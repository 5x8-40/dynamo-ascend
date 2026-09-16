// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Public request-classifier contract and startup configuration.

mod config;
mod registry;

pub(crate) use config::RawRequestClassifierConfig;
pub use config::RequestClassifierConfig;
pub(crate) use registry::RequestClassifierRegistry;
pub use registry::{
    RequestClassifierFactory, RequestClassifierParameters, RequestClassifierProvider,
    RequestClassifierProviderError, RequestClassifierRegistryError,
};

use std::error::Error;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use async_trait::async_trait;
use tokio::time::Instant;

use crate::protocols::WorkerWithDpRank;
use crate::scheduling::{SessionContext, policy_queue::QueueSnapshot};

#[derive(Debug)]
pub struct ClassifyRequest {
    pub(crate) classification_id: u64,
    request_id: Option<String>,
    policy_class: Option<String>,
    pub(crate) overrides: ClassificationOverrides,
    ingress_at: Instant,
    input_tokens: usize,
    initial_cached_tokens: usize,
    session_context: Option<SessionContext>,
}

#[derive(Clone, Debug, Default)]
pub(crate) struct ClassificationOverrides {
    policy_class: Option<String>,
    due_at: Option<Instant>,
    scheduling_cost_tokens: Option<usize>,
}

impl ClassifyRequest {
    #[cfg(test)]
    pub(crate) fn new(input_tokens: usize, initial_cached_tokens: usize) -> Self {
        Self::with_timing(input_tokens, initial_cached_tokens, Instant::now())
    }

    pub(crate) fn with_timing(
        input_tokens: usize,
        initial_cached_tokens: usize,
        ingress_at: Instant,
    ) -> Self {
        Self {
            classification_id: 0,
            request_id: None,
            policy_class: None,
            overrides: ClassificationOverrides::default(),
            ingress_at,
            input_tokens,
            initial_cached_tokens,
            session_context: None,
        }
    }

    pub(crate) fn with_request_id(mut self, request_id: impl Into<String>) -> Self {
        self.request_id = Some(request_id.into());
        self
    }

    pub(crate) fn with_initial_policy_class(mut self, policy_class: impl Into<String>) -> Self {
        self.policy_class = Some(policy_class.into());
        self
    }

    pub(crate) fn with_session_context(mut self, session_context: SessionContext) -> Self {
        self.session_context = Some(session_context);
        self
    }

    pub fn request_id(&self) -> Option<&str> {
        self.request_id.as_deref()
    }

    pub fn policy_class(&self) -> Option<&str> {
        self.overrides
            .policy_class
            .as_deref()
            .or(self.policy_class.as_deref())
    }

    pub fn set_policy_class(&mut self, policy_class: impl Into<String>) {
        self.overrides.policy_class = Some(policy_class.into());
    }

    pub fn input_tokens(&self) -> usize {
        self.input_tokens
    }

    /// Return the original router ingress time on the monotonic clock.
    pub fn ingress_at(&self) -> Instant {
        self.ingress_at
    }

    pub fn due_at(&self) -> Option<Instant> {
        self.overrides.due_at
    }

    pub fn set_due_at(&mut self, due_at: Instant) {
        self.overrides.due_at = Some(due_at);
    }

    pub fn scheduling_cost_tokens(&self) -> usize {
        self.overrides.scheduling_cost_tokens.unwrap_or_else(|| {
            QueueSnapshot::new(self.input_tokens, self.initial_cached_tokens).scheduling_cost_tokens
        })
    }

    pub fn set_scheduling_cost_tokens(&mut self, scheduling_cost_tokens: usize) {
        self.overrides.scheduling_cost_tokens = Some(scheduling_cost_tokens);
    }

    pub fn session_context(&self) -> Option<&SessionContext> {
        self.session_context.as_ref()
    }

    /// Only the explicit overrides feed the queue: cache eligibility is
    /// recomputed from the current workers at enqueue, because worker state
    /// may have changed while the classification was pending.
    pub(crate) fn into_queue_inputs(self) -> (Option<String>, Option<Instant>, Option<usize>) {
        (
            self.overrides.policy_class,
            self.overrides.due_at,
            self.overrides.scheduling_cost_tokens,
        )
    }
}

/// Error returned by [`RequestClassifier::classify`].
pub type ClassifierError = dyn Error + Send + Sync + 'static;

/// Cause delivered to the classifier when a request aborts. Produced by the
/// router or the worker path, not by the classifier.
pub type AbortCause = dyn Error + Send + Sync + 'static;

/// Request lifecycle events, delivered to [`RequestClassifier::on_event`] one
/// at a time in lifecycle order. A terminal event can arrive without a prior
/// `Sent`: an attempt that recorded a worker via selection but was never
/// dispatched still ends with `Completed`.
///
/// Delivery is asynchronous, but a re-registered request id is not classified
/// until the previous lifecycle's terminal event has been delivered, so a
/// plugin keying state by request id observes each lifecycle's `classify` and
/// events in order and never receives a stale terminal event after the id's
/// next `classify`.
#[derive(Debug)]
#[non_exhaustive]
pub enum ClassifyEvent {
    Sent {
        request_id: String,
        worker: WorkerWithDpRank,
    },
    Responding {
        request_id: String,
        worker: WorkerWithDpRank,
    },
    Completed {
        request_id: String,
        worker: WorkerWithDpRank,
        context_tokens: Option<usize>,
    },
    Aborted {
        request_id: String,
        worker: Option<WorkerWithDpRank>,
        error: Option<Arc<AbortCause>>,
    },
}

/// An independently pollable classifier invocation. See [`RequestClassifier`]
/// for why this is a `'static` future rather than `async fn`.
pub type ClassifyFuture =
    Pin<Box<dyn Future<Output = Result<ClassifyRequest, Box<ClassifierError>>> + Send + 'static>>;

/// User-provided request classifier.
///
/// [`Self::classify`] deliberately avoids `async fn` and returns a `'static`
/// future which the router polls on its own runtime, so a classifier may
/// await, sleep, or park a request without blocking a thread and without
/// standing up a runtime of its own. `async fn` cannot express this: its
/// future borrows `&mut self` for the whole pending wait, which would hold the
/// router-wide classifier lock and block every other request and every event.
///
/// Because the returned future is `'static`, state it touches must be owned or
/// shared rather than borrowed from `self`:
///
/// ```ignore
/// struct Pauser {
///     state: Arc<Mutex<ProgramState>>,
/// }
///
/// impl RequestClassifier for Pauser {
///     fn classify(&mut self, request: ClassifyRequest) -> ClassifyFuture {
///         // Clone the handle into the future; `self` cannot be borrowed.
///         let state = Arc::clone(&self.state);
///         Box::pin(async move {
///             state.lock().await.wait_for_slot().await;
///             Ok(request)
///         })
///     }
/// }
/// ```
///
/// `classify`'s synchronous prologue runs under the router-wide classifier
/// lock: build the future and return promptly, then wait inside it. Blocking
/// before the future is returned stalls classification and event delivery.
///
/// [`Self::on_event`] is `async fn`. A dedicated router task delivers events
/// one at a time in lifecycle order and awaits each callback, so `on_event`
/// may await freely. A pending `classify` future never blocks delivery, but a
/// slow `on_event` delays classification of new requests because both share
/// the classifier lock. Terminal events arrive asynchronously, shortly after
/// the request ends; events still queued at router shutdown are dropped.
#[async_trait]
pub trait RequestClassifier: Send + 'static {
    fn classify(&mut self, request: ClassifyRequest) -> ClassifyFuture {
        Box::pin(async move { Ok(request) })
    }

    async fn on_event(&mut self, _event: ClassifyEvent) {}
}
