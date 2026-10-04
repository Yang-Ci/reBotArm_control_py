
// REBOTARM_DM_TIMED_ABI
#[unsafe(no_mangle)]
pub extern "C" fn rebotarm_dm_serial_split_v1() -> i32 { 1 }

#[unsafe(no_mangle)]
pub extern "C" fn rebotarm_dm_send_batch(
    handles: *const *mut MotorHandle, commands: *const f32, count: u32, mode: u32,
) -> i32 {
    if handles.is_null() || commands.is_null() || count == 0 || count > 64 || !(1..=3).contains(&mode) {
        set_last_error("invalid DM batch arguments"); return -1;
    }
    let handles = unsafe { std::slice::from_raw_parts(handles, count as usize) };
    let values = unsafe { std::slice::from_raw_parts(commands, count as usize * 5) };
    let mut targets = Vec::with_capacity(count as usize);
    for (handle, target) in handles.iter().zip(values.chunks_exact(5)) {
        let guard = lock_motor_inner!(*handle, "null DM batch motor");
        match &*guard {
            MotorHandleInner::Damiao(motor) => {
                targets.push((Arc::clone(motor), <[f32; 5]>::try_from(target).unwrap()));
            }
            _ => { set_last_error("DM batch requires Damiao motors"); return -1; }
        }
    }
    match DamiaoMotor::send_group(&targets, mode) {
        Ok(()) => 0,
        Err(error) => { set_last_error(error.to_string()); -1 }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn rebotarm_dm_get_state_timed(
    motor: *mut MotorHandle, out_state: *mut MotorState,
    out_sequence: *mut u64, out_age: *mut f64, out_limits: *mut f32,
) -> i32 {
    if motor.is_null() || out_state.is_null() || out_sequence.is_null()
        || out_age.is_null() || out_limits.is_null() {
        set_last_error("null DM timed-state argument");
        return -1;
    }
    let owned = {
        let guard = lock_motor_inner!(motor, "motor is null");
        match &*guard {
            MotorHandleInner::Damiao(m) => Arc::clone(m),
            _ => { set_last_error("DM timed state requires a Damiao motor"); return -1; }
        }
    };
    let limits = match motor_vendor_damiao::model_limits(&owned.model) {
        Some(value) => value,
        None => { set_last_error("unknown DM model limits"); return -1; }
    };
    let snapshot = match owned.latest_state_timed() {
        Ok(value) => value,
        Err(error) => { set_last_error(error.to_string()); return -1; }
    };
    unsafe {
        *out_limits = limits.0;
        *out_limits.add(1) = limits.1;
        *out_limits.add(2) = limits.2;
        *out_state = MotorState::default();
        *out_sequence = 0;
        *out_age = f64::INFINITY;
        if let Some((state, sequence, age)) = snapshot {
            *out_state = MotorState {
                has_value: 1, can_id: state.can_id, arbitration_id: state.arbitration_id,
                status_code: state.status_code, pos: state.pos, vel: state.vel,
                torq: state.torq, t_mos: state.t_mos, t_rotor: state.t_rotor,
            };
            *out_sequence = sequence;
            *out_age = age.as_secs_f64();
        }
    }
    0
}

#[cfg(test)]
mod rebotarm_dm_state_tests {
    use super::*;
    use motor_core::bus::CanFrame;
    use motor_core::MotorDevice;
    struct MockBus;
    impl CanBus for MockBus {
        fn send(&self, _: CanFrame) -> motor_core::error::Result<()> { Ok(()) }
        fn recv(&self, _: Duration) -> motor_core::error::Result<Option<CanFrame>> { Ok(None) }
        fn shutdown(&self) -> motor_core::error::Result<()> { Ok(()) }
    }

    #[derive(Default)]
    struct CaptureBus { frames: Mutex<Vec<CanFrame>>, batches: std::sync::atomic::AtomicUsize }
    impl CanBus for CaptureBus {
        fn send(&self, frame: CanFrame) -> motor_core::error::Result<()> {
            self.frames.lock().unwrap().push(frame); Ok(())
        }
        fn send_batch(&self, frames: &[CanFrame]) -> motor_core::error::Result<()> {
            self.batches.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            self.frames.lock().unwrap().extend_from_slice(frames); Ok(())
        }
        fn recv(&self, _: Duration) -> motor_core::error::Result<Option<CanFrame>> { Ok(None) }
        fn shutdown(&self) -> motor_core::error::Result<()> { Ok(()) }
    }

    #[test]
    fn batch_matches_vendor_encoding_and_validates_all_commands_before_send() {
        let bus_impl = Arc::new(CaptureBus::default());
        let bus: Arc<dyn CanBus> = bus_impl.clone();
        let motors: Vec<_> = (1..=6).map(|id|
            Arc::new(DamiaoMotor::new(id, id + 0x10, "4340P", bus.clone()).unwrap())).collect();
        let handles: Vec<_> = motors.iter().map(|m| Box::into_raw(Box::new(MotorHandle {
            inner: Mutex::new(MotorHandleInner::Damiao(m.clone())),
        }))).collect();
        let values: Vec<f32> = (0..6).flat_map(|i| [i as f32 * 0.1, 0.03, 120., 8., 2.]).collect();
        for mode in [1, 2, 3] {
            assert_eq!(rebotarm_dm_send_batch(handles.as_ptr(), values.as_ptr(), 6, mode), 0);
            let sent = bus_impl.frames.lock().unwrap().drain(..).collect::<Vec<_>>();
            assert_eq!(sent.len(), 6);
            for (motor, target) in motors.iter().zip(values.chunks_exact(5)) {
                if mode == 1 { motor.send_cmd_mit(target[0], target[1], target[2], target[3], target[4]).unwrap(); }
                else if mode == 2 { motor.send_cmd_pos_vel(target[0], target[1]).unwrap(); }
                else { motor.request_motor_feedback().unwrap(); }
            }
            let expected = bus_impl.frames.lock().unwrap().drain(..).collect::<Vec<_>>();
            for (actual, expected) in sent.iter().zip(expected) {
                assert_eq!(actual.arbitration_id, expected.arbitration_id);
                assert_eq!(actual.data, expected.data);
            }
        }
        assert_eq!(bus_impl.batches.load(std::sync::atomic::Ordering::Relaxed), 3);
        let mut invalid = values.clone();
        invalid[29] = f32::NAN;
        assert_eq!(rebotarm_dm_send_batch(handles.as_ptr(), invalid.as_ptr(), 6, 1), -1);
        assert!(bus_impl.frames.lock().unwrap().is_empty());
        let other: Arc<dyn CanBus> = Arc::new(MockBus);
        let cross_bus = Arc::new(DamiaoMotor::new(7, 0x17, "4310", other).unwrap());
        assert!(DamiaoMotor::send_group(&[(motors[0].clone(), [0., 0., 1., 1., 0.]),
            (cross_bus, [0., 0., 1., 1., 0.])], 1).is_err());
        assert!(bus_impl.frames.lock().unwrap().is_empty());
        for handle in handles { unsafe { drop(Box::from_raw(handle)); } }
    }

    #[test]
    fn only_sensor_packets_advance_the_coherent_timed_snapshot() {
        let bus: Arc<dyn CanBus> = Arc::new(MockBus);
        let motor = Arc::new(DamiaoMotor::new(1, 0x11, "4340P", bus).unwrap());
        let handle = Box::into_raw(Box::new(MotorHandle {
            inner: Mutex::new(MotorHandleInner::Damiao(Arc::clone(&motor))),
        }));
        let mut state = MotorState::default();
        let mut seq = 0;
        let mut age = 0.0;
        let mut limits = [0.0f32; 3];
        let read = |state: &mut MotorState, seq: &mut u64, age: &mut f64, limits: &mut [f32; 3]| {
            assert_eq!(rebotarm_dm_get_state_timed(handle, state, seq, age, limits.as_mut_ptr()), 0);
        };
        read(&mut state, &mut seq, &mut age, &mut limits);
        assert_eq!(state.has_value, 0);
        assert_eq!(seq, 0);
        assert!(age.is_infinite());
        assert_eq!(limits, [12.5, 10.0, 28.0]);
        let sensor = CanFrame { arbitration_id: 0x11, dlc: 8, is_extended: false,
            is_rx: true, data: [0x11, 0x80, 0, 0x80, 0, 0, 25, 25] };
        motor.process_feedback_frame(sensor).unwrap();
        read(&mut state, &mut seq, &mut age, &mut limits);
        assert_eq!(seq, 1);
        assert_eq!(state.status_code, 1);
        let first_age = age;
        std::thread::sleep(Duration::from_millis(5));
        // A register reply must not refresh the position timestamp.
        motor.process_feedback_frame(CanFrame { data: [1, 0, 0x33, 21, 0, 0, 0x48, 0x41], ..sensor }).unwrap();
        read(&mut state, &mut seq, &mut age, &mut limits);
        assert_eq!(seq, 1);
        assert!(age >= first_age + 0.004);
        assert!(motor.process_feedback_frame(CanFrame { dlc: 4, ..sensor }).is_err());
        // Identical positions still count as new sensor packets.
        motor.process_feedback_frame(sensor).unwrap();
        read(&mut state, &mut seq, &mut age, &mut limits);
        assert_eq!(seq, 2);
        assert!(age < 0.01);
        unsafe { drop(Box::from_raw(handle)); }
    }
}
