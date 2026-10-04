
#[cfg(test)]
mod rebotarm_dm_serial_tests {
    use super::*;
    use std::sync::{Arc, mpsc};

    struct TestPort {
        read_started: mpsc::Sender<()>,
        release: Arc<Mutex<mpsc::Receiver<()>>>,
        written: Arc<Mutex<Vec<u8>>>,
    }
    impl Read for TestPort {
        fn read(&mut self, _: &mut [u8]) -> std::io::Result<usize> {
            self.read_started.send(()).unwrap();
            self.release.lock().unwrap().recv_timeout(Duration::from_secs(2)).unwrap();
            Err(std::io::Error::from(std::io::ErrorKind::TimedOut))
        }
    }
    impl Write for TestPort {
        fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
            // Force write_all to perform multiple writes to detect interleaving.
            let count = bytes.len().min(3);
            self.written.lock().unwrap().extend_from_slice(&bytes[..count]);
            std::thread::yield_now();
            Ok(count)
        }
        fn flush(&mut self) -> std::io::Result<()> { Ok(()) }
    }
    impl SerialPort for TestPort {
        fn name(&self) -> Option<String> { None }
        fn baud_rate(&self) -> serialport::Result<u32> { Ok(921600) }
        fn data_bits(&self) -> serialport::Result<DataBits> { Ok(DataBits::Eight) }
        fn flow_control(&self) -> serialport::Result<FlowControl> { Ok(FlowControl::None) }
        fn parity(&self) -> serialport::Result<Parity> { Ok(Parity::None) }
        fn stop_bits(&self) -> serialport::Result<StopBits> { Ok(StopBits::One) }
        fn timeout(&self) -> Duration { Duration::from_millis(10) }
        fn set_baud_rate(&mut self, _: u32) -> serialport::Result<()> { Ok(()) }
        fn set_data_bits(&mut self, _: DataBits) -> serialport::Result<()> { Ok(()) }
        fn set_flow_control(&mut self, _: FlowControl) -> serialport::Result<()> { Ok(()) }
        fn set_parity(&mut self, _: Parity) -> serialport::Result<()> { Ok(()) }
        fn set_stop_bits(&mut self, _: StopBits) -> serialport::Result<()> { Ok(()) }
        fn set_timeout(&mut self, _: Duration) -> serialport::Result<()> { Ok(()) }
        fn write_request_to_send(&mut self, _: bool) -> serialport::Result<()> { Ok(()) }
        fn write_data_terminal_ready(&mut self, _: bool) -> serialport::Result<()> { Ok(()) }
        fn read_clear_to_send(&mut self) -> serialport::Result<bool> { Ok(true) }
        fn read_data_set_ready(&mut self) -> serialport::Result<bool> { Ok(true) }
        fn read_ring_indicator(&mut self) -> serialport::Result<bool> { Ok(false) }
        fn read_carrier_detect(&mut self) -> serialport::Result<bool> { Ok(true) }
        fn bytes_to_read(&self) -> serialport::Result<u32> { Ok(1) }
        fn bytes_to_write(&self) -> serialport::Result<u32> { Ok(0) }
        fn clear(&self, _: serialport::ClearBuffer) -> serialport::Result<()> { Ok(()) }
        fn try_clone(&self) -> serialport::Result<Box<dyn SerialPort>> {
            Ok(Box::new(Self { read_started: self.read_started.clone(),
                release: self.release.clone(), written: self.written.clone() }))
        }
        fn set_break(&self) -> serialport::Result<()> { Ok(()) }
        fn clear_break(&self) -> serialport::Result<()> { Ok(()) }
    }

    #[test]
    fn blocked_receive_allows_transmit_and_writes_keep_complete_can_frames() {
        let (started_tx, started_rx) = mpsc::channel();
        let (release_tx, release_rx) = mpsc::channel();
        let written = Arc::new(Mutex::new(Vec::new()));
        let port = TestPort { read_started: started_tx,
            release: Arc::new(Mutex::new(release_rx)), written: written.clone() };
        let bus = Arc::new(DmSerialBus { writer: Mutex::new(port.try_clone().unwrap()),
            inner: Mutex::new(Inner { port: Box::new(port), rx_buf: VecDeque::new() }) });
        let reader_bus = bus.clone();
        let reader = std::thread::spawn(move || reader_bus.recv(Duration::ZERO));
        started_rx.recv_timeout(Duration::from_secs(1)).unwrap();
        let (done_tx, done_rx) = mpsc::channel();
        let handles: Vec<_> = (1..=7).map(|id| {
            let sender = bus.clone();
            let done = done_tx.clone();
            std::thread::spawn(move || {
                let frame = CanFrame { arbitration_id: id, data: [id as u8; 8],
                    dlc: 8, is_extended: false, is_rx: false };
                sender.send(frame).unwrap();
                done.send(()).unwrap();
            })
        }).collect();
        let completed = (0..7).all(|_| done_rx.recv_timeout(Duration::from_millis(100)).is_ok());
        release_tx.send(()).unwrap();
        reader.join().unwrap().unwrap();
        for handle in handles { handle.join().unwrap(); }
        assert!(completed, "TX waited on the RX lock");
        let bytes = written.lock().unwrap();
        assert_eq!(bytes.len(), 7 * TX_FRAME_LEN);
        for packet in bytes.chunks_exact(TX_FRAME_LEN) {
            assert_eq!(&packet[..4], &[0x55, 0xaa, 0x1e, 0x03]);
            let id = packet[13];
            assert_eq!(&packet[21..29], &[id; 8]);
        }
    }

    #[test]
    fn concurrent_batches_keep_group_packets_together_and_reject_invalid_frames() {
        let (started_tx, _) = mpsc::channel();
        let (_, release_rx) = mpsc::channel();
        let written = Arc::new(Mutex::new(Vec::new()));
        let port = TestPort { read_started: started_tx,
            release: Arc::new(Mutex::new(release_rx)), written: written.clone() };
        let bus = Arc::new(DmSerialBus { writer: Mutex::new(port.try_clone().unwrap()),
            inner: Mutex::new(Inner { port: Box::new(port), rx_buf: VecDeque::new() }) });
        let handles: Vec<_> = [1, 4].iter().map(|start| {
            let sender = bus.clone();
            let frames: Vec<_> = (*start..*start + 3).map(|id| CanFrame {
                arbitration_id: id, data: [id as u8; 8], dlc: 8, is_extended: false, is_rx: false,
            }).collect();
            std::thread::spawn(move || sender.send_batch(&frames).unwrap())
        }).collect();
        for handle in handles { handle.join().unwrap(); }
        let bytes = written.lock().unwrap();
        let ids: Vec<_> = bytes.chunks_exact(TX_FRAME_LEN).map(|p| p[13]).collect();
        assert!(ids == [1, 2, 3, 4, 5, 6] || ids == [4, 5, 6, 1, 2, 3]);
        for packet in bytes.chunks_exact(TX_FRAME_LEN) { assert_eq!(&packet[21..29], &[packet[13]; 8]); }
        let length = bytes.len();
        drop(bytes);
        let valid = CanFrame { arbitration_id: 1, data: [0; 8], dlc: 8, is_extended: false, is_rx: false };
        assert!(bus.send_batch(&[valid, CanFrame { dlc: 9, ..valid }]).is_err());
        assert_eq!(written.lock().unwrap().len(), length);
    }
}
