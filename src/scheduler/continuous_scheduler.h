#pragma once

#include <absl/time/time.h>
#include <folly/MPMCQueue.h>

#include <memory>
#include <queue>

#include "common/macros.h"
#include "engine/batch.h"
#include "memory/block_manager.h"
#include "request/request.h"
#include "request/sequence.h"
#include "response_handler.h"
#include "scheduler.h"

namespace llm {
class Engine;

// TODO: add schedule config to control the max number of tokens per batch, max
// number of seqs per batch and the time out value.
class ContinuousScheduler final : public Scheduler {
 public:
  struct Options {
    // the maximum number of tokens per batch
    DEFINE_ARG(int32_t, max_tokens_per_batch) = 256;

    // the maximum number of sequences per batch
    DEFINE_ARG(int32_t, max_seqs_per_batch) = 64;

    // the number of speculative tokens per step
    DEFINE_ARG(int32_t, num_speculative_tokens) = 0;
  };

  ContinuousScheduler(Engine* engine, const Options& options);

  ~ContinuousScheduler();

  // schedule a request, thread safe and non-blocking
  // may return false if the queue is full
  bool schedule(std::unique_ptr<Request>& request) override;

  // step the scheduler forward by one step
  // may get blocked if there are no requests to process
  void step(const absl::Duration& timeout) override;

  // run the scheduler until all pending + scheduled requests are completed
  void run_until_complete() override;

  // inc/dec pending requests
  void inc_pending_requests(size_t count) override {
    pending_requests_.fetch_add(count, std::memory_order_relaxed);
  }
  void dec_pending_requests() override {
    const auto old_value =
        pending_requests_.fetch_sub(1, std::memory_order_relaxed);
    CHECK_GT(old_value, 0) << "pending requests underflow";
  }

 private:
  Batch wait_for_batch(const absl::Duration& timeout);

  // build a batch of requests from the priority queue
  Batch build_sequence_batch();

  // process the batch output
  void process_batch_output();

  // allocate blocks for a sequence, honoring the tokens budget.
  // * for prefill sequence, the allocated_tokens will be within
  // [1, num_prompt_tokens - num_tokens_in_kv_cache].
  // * for decode sequence, the actual_tokens usually would be 1 or K for
  // speculative decoding.
  // returns false if no blocks can be allocated.
  bool allocate_blocks_for(Sequence* sequence,
                           size_t token_budget,
                           size_t* actual_tokens);

  const Options options_;

  // the engine to run the batch
  Engine* engine_;

  // the block manager to manage the cache blocks
  BlockManager* block_manager_;

  // tokenizer
  std::unique_ptr<Tokenizer> tokenizer_;

  // a thread safe queue of requests, bounded by kRequestQueueSize
  // the schedule owns the requests and manages their lifetimes.
  folly::MPMCQueue<Request*> request_queue_;

  // Requests with HIGH priority are processed first, followed by MEDIUM
  // priority requests, and finally LOW priority requests. Within each priority
  // level, requests are handled on First-Come-First-Served (FCFS) basis.
  using MinHeap =
      std::priority_queue<Request*, std::vector<Request*>, RequestPtrGreater>;
  MinHeap priority_queue_;

  // a batch of requests in running state, sorted by priority from high to low.
  std::vector<Request*> running_requests_;

  // a batch of sequences that scheduled to run, sorted by priority from high to
  std::vector<Sequence*> running_sequences_;

  // token budget for each running sequence
  std::vector<size_t> running_sequences_budgets_;

  // preemptable requests that hold cache slots, sorted by priority from high to
  // low.
  std::deque<Request*> preemptable_requests_;

  std::unique_ptr<ResponseHandler> response_handler_;

  bool enable_prefix_cache_ = false;

  // the number of requests that are waiting to be scheduled
  std::atomic<size_t> pending_requests_{0};
};

}  // namespace llm
