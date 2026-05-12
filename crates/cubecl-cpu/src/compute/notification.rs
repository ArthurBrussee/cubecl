/// Struct used inside the CPUExecutionQueue to send the flush command
use std::{
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    thread::{self, Thread},
};

#[derive(Clone)]
pub(super) struct Notification {
    ready: Arc<AtomicBool>,
    current_thread: Thread,
}

const MAX_SPIN_ITERATION: u16 = 8192;

impl Notification {
    #[inline]
    pub fn new() -> Self {
        let ready = Arc::new(AtomicBool::new(false));
        let current_thread = thread::current();
        Self {
            ready,
            current_thread,
        }
    }

    #[inline]
    pub fn send(&self) {
        self.ready.store(true, Ordering::Release);
        self.current_thread.unpark();
    }

    #[inline]
    pub fn wait(&self) {
        for _ in 0..MAX_SPIN_ITERATION {
            if self.ready.load(Ordering::Acquire) {
                return;
            }
            std::hint::spin_loop();
        }

        while !self.ready.load(Ordering::Acquire) {
            std::thread::park();
        }
    }
}
