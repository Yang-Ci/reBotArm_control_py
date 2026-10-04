
#[cfg(test)]
mod rebotarm_feedback_lock_tests {
    use super::*;
    use crate::motor_control_ffi::motor_handle_send_mit;
    use motor_core::bus::{CanBus, CanFrame};
    use std::sync::mpsc;
    use std::time::Instant;

    struct TestBus { requested: mpsc::Sender<()> }
    impl CanBus for TestBus {
        fn send(&self, frame: CanFrame) -> motor_core::error::Result<()> {
            if ((frame.arbitration_id >> 24) & 0x1f) == 17 {
                let _ = self.requested.send(());
            }
            Ok(())
        }
        fn recv(&self, _: Duration) -> motor_core::error::Result<Option<CanFrame>> { Ok(None) }
        fn shutdown(&self) -> motor_core::error::Result<()> { Ok(()) }
    }

    #[test]
    fn send_progresses_while_same_handle_waits_for_parameter() {
        let (tx, rx) = mpsc::channel();
        let bus: Arc<dyn CanBus> = Arc::new(TestBus { requested: tx });
        let motor = Arc::new(RobstrideMotor::new(1, 0xfd, "rs-06", bus).unwrap());
        let handle = Box::into_raw(Box::new(MotorHandle {
            inner: Mutex::new(MotorHandleInner::Robstride(motor)),
        }));
        let address = handle as usize;
        let reader = std::thread::spawn(move || {
            let mut out = 0.0f32;
            motor_handle_robstride_get_param_f32_host_id(address as *mut MotorHandle, 0x7019, 0xfd, 300, &mut out)
        });
        rx.recv_timeout(Duration::from_secs(1)).unwrap();
        let began = Instant::now();
        let rc = motor_handle_send_mit(handle, 0.0, 0.01, 10.0, 1.0, 0.0);
        let elapsed = began.elapsed();
        let read_rc = reader.join().unwrap();
        unsafe { drop(Box::from_raw(handle)); }
        assert_eq!(rc, 0);
        assert_eq!(read_rc, -1); // No reply: timeout must not look like fresh data.
        assert!(elapsed < Duration::from_millis(100), "send blocked for {:?}", elapsed);
    }
}
