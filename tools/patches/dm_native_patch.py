"""Reviewed, idempotent changes to motorbridge 0.5.6; no binary layout guessing."""
from pathlib import Path


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise RuntimeError(f"Upstream DM source changed near {old[:70]!r}")
    return text.replace(old, new, 1)


def patch_dm(source: Path):
    path = source / "motor_vendors/damiao/src/motor.rs"
    text = path.read_text(encoding="utf-8")
    if "pub fn latest_state_timed" not in text:
        text = replace_once(text, "{AtomicBool, Ordering}", "{AtomicBool, AtomicU64, Ordering}")
        text = replace_once(text, "    state: Mutex<Option<(MotorFeedbackState, Instant)>>,",
                            "    state: Mutex<Option<(MotorFeedbackState, Instant)>>,\n    feedback_sequence: AtomicU64,")
        text = replace_once(text, "            state: Mutex::new(None),",
                            "            state: Mutex::new(None),\n            feedback_sequence: AtomicU64::new(0),")
        text = replace_once(text, "    pub fn request_fresh_state(&self, timeout: Duration)", '''    // Sequence changes only on a sensor packet. Reading the cache or receiving
    // register replies cannot make old feedback look fresh. One lock makes
    // state, timestamp and sequence a coherent snapshot.
    pub fn latest_state_timed(&self) -> Result<Option<(MotorFeedbackState, u64, Duration)>> {
        let state = self.state.lock()
            .map_err(|_| MotorError::Io("state lock poisoned".to_string()))?;
        Ok(state.as_ref().map(|(value, received)|
            (*value, self.feedback_sequence.load(Ordering::Relaxed), received.elapsed())))
    }

    pub fn request_fresh_state(&self, timeout: Duration)''')
        text = replace_once(text, '''        self.state
            .lock()
            .map_err(|_| MotorError::Io("state lock poisoned".to_string()))?
            .replace((state, Instant::now()));''', '''        let mut cached = self.state.lock()
            .map_err(|_| MotorError::Io("state lock poisoned".to_string()))?;
        cached.replace((state, Instant::now()));
        self.feedback_sequence.fetch_add(1, Ordering::Relaxed);''')
        text = replace_once(text, "    fn process_feedback_frame_impl(&self, frame: CanFrame) -> Result<()> {",
                            '''    fn process_feedback_frame_impl(&self, frame: CanFrame) -> Result<()> {
        if frame.dlc != 8 {
            return Err(MotorError::Protocol("DM feedback requires eight bytes".to_string()));
        }''')
        path.write_text(text, encoding="utf-8")

    path = source / "motor_core/src/bus.rs"
    text = path.read_text(encoding="utf-8")
    if "fn send_batch(&self" not in text:
        text = replace_once(text, "    fn send(&self, frame: CanFrame) -> Result<()>;", '''    fn send(&self, frame: CanFrame) -> Result<()>;
    fn send_batch(&self, frames: &[CanFrame]) -> Result<()> {
        for frame in frames { self.send(*frame)?; }
        Ok(())
    }''')
        path.write_text(text, encoding="utf-8")

    path = source / "motor_core/src/controller.rs"
    text = path.read_text(encoding="utf-8")
    if "fn send_batch(&self" not in text:
        anchor = "    fn recv(&self, timeout: Duration) -> Result<Option<crate::bus::CanFrame>> {"
        text = replace_once(text, anchor, '''    fn send_batch(&self, frames: &[crate::bus::CanFrame]) -> Result<()> {
        if self.min_gap_ns.load(Ordering::Acquire) == 0 {
            return self.inner.send_batch(frames);
        }
        // Preserve an explicitly selected per-frame pacing policy.
        for frame in frames { self.send(*frame)?; }
        Ok(())
    }

''' + anchor)
        path.write_text(text, encoding="utf-8")

    path = source / "motor_core/src/dm_serial.rs"
    text = path.read_text(encoding="utf-8")
    if "writer: Mutex<Box<dyn SerialPort>>" not in text:
        text = replace_once(text, "pub struct DmSerialBus {\n    inner: Mutex<Inner>,",
                            "pub struct DmSerialBus {\n    inner: Mutex<Inner>,\n    writer: Mutex<Box<dyn SerialPort>>,")
        text = replace_once(text, "        Ok(Self {\n            inner: Mutex::new(Inner {", '''        // Clone the already-open port; no second adapter open. Independent
        // read/write locks allow full duplex without interleaving CAN packets.
        let writer = port_obj.try_clone()
            .map_err(|e| MotorError::Io(format!("clone serial port failed: {e}")))?;
        Ok(Self {
            writer: Mutex::new(writer),
            inner: Mutex::new(Inner {''')
        start = text.index("    fn send(&self, frame: CanFrame)")
        stop = text.index("    fn recv(&self", start)
        send = text[start:stop].replace("let mut inner", "let mut writer").replace(".inner", ".writer").replace("        inner\n            .port\n", "        writer\n")
        text = text[:start] + send + text[stop:]
        start = text.index("    fn shutdown(&self)")
        tail = text[start:].replace("let mut inner", "let mut writer").replace(".inner", ".writer").replace("        inner\n            .port\n", "        writer\n")
        text = text[:start] + tail
        path.write_text(text, encoding="utf-8")
    text = path.read_text(encoding="utf-8")
    if "fn send_batch(&self" not in text:
        # Do not change the bridge protocol: concatenate complete 30-byte
        # packets in one write_all. Invalid frames cause no partial batch.
        anchor = "    fn recv(&self, timeout: Duration) -> Result<Option<CanFrame>> {"
        text = replace_once(text, anchor, '''    fn send_batch(&self, frames: &[CanFrame]) -> Result<()> {
        let mut raw = Vec::with_capacity(frames.len() * TX_FRAME_LEN);
        for frame in frames { raw.extend_from_slice(&Self::encode_tx(*frame)?); }
        self.writer.lock()
            .map_err(|_| MotorError::Io("dm-serial TX lock poisoned".to_string()))?
            .write_all(&raw)
            .map_err(|e| MotorError::Io(format!("dm-serial batch write failed: {e}")))
    }

''' + anchor)
        path.write_text(text, encoding="utf-8")

    path = source / "motor_vendors/damiao/src/motor.rs"
    text = path.read_text(encoding="utf-8")
    if "    pub fn send_group" in text:
        start = text.index("    pub fn send_group")
        stop = text.index("    pub fn send_cmd_mit", start)
        text = text[:start] + text[stop:]
    if "pub fn send_group" not in text:
        anchor = "    pub fn send_cmd_mit("
        text = replace_once(text, anchor, '''    pub fn send_group(commands: &[(Arc<Self>, [f32; 5])], mode: u32) -> Result<()> {
        let Some((first, _)) = commands.first() else { return Ok(()); };
        let mut frames = Vec::with_capacity(commands.len());
        for (motor, command) in commands {
            if !Arc::ptr_eq(&first.bus, &motor.bus) {
                return Err(MotorError::InvalidArgument("DM batch spans different buses".to_string()));
            }
            if command.iter().any(|v| !v.is_finite()) {
                return Err(MotorError::InvalidArgument("nonfinite DM batch command".to_string()));
            }
            let (id, data) = if mode == 3 {
                (0x7ff, encode_feedback_request_cmd(motor.motor_id))
            } else if mode == 2 {
                if command[1] <= 0.0 {
                    return Err(MotorError::InvalidArgument("positive POS_VEL limit required".to_string()));
                }
                (u32::from(0x100u16 + motor.motor_id), encode_pos_vel_cmd(command[0], command[1]))
            } else {
                let limits = Limits { p_min: motor.limits.p_min, p_max: motor.limits.p_max,
                    v_min: motor.limits.v_min, v_max: motor.limits.v_max,
                    t_min: motor.limits.t_min, t_max: motor.limits.t_max };
                (u32::from(motor.motor_id), encode_mit_cmd(command[0], command[1], command[4],
                    command[2], command[3], limits))
            };
            frames.push(CanFrame { arbitration_id: id, data, dlc: 8, is_extended: false, is_rx: false });
        }
        first.bus.send_batch(&frames)
    }

''' + anchor)
        path.write_text(text, encoding="utf-8")
    # Always refresh our appended modules on an idempotent rebuild.
    for relative, template, marker in (
        ("motor_abi/src/state_ffi.rs", "dm_state_ffi.rs", "\n// REBOTARM_DM_TIMED_ABI"),
        ("motor_core/src/dm_serial.rs", "dm_serial_test.rs", "\n#[cfg(test)]\nmod rebotarm_dm_serial_tests"),
    ):
        path = source / relative
        text = path.read_text(encoding="utf-8").split(marker)[0]
        path.write_text(text + (Path(__file__).parent / template).read_text(encoding="utf-8"), encoding="utf-8")
