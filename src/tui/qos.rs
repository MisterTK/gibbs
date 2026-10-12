#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QosClass {
    UserInteractive,
    UserInitiated,
    Default,
    Utility,
    Background,
}

#[cfg(target_os = "macos")]
pub fn set_qos(class: QosClass) -> bool {
    extern "C" {
        fn pthread_set_qos_class_self_np(qos_class: u32, relative_priority: i32) -> i32;
    }
    let code: u32 = match class {
        QosClass::UserInteractive => 0x21,
        QosClass::UserInitiated => 0x19,
        QosClass::Default => 0x15,
        QosClass::Utility => 0x11,
        QosClass::Background => 0x09,
    };
    let res = unsafe { pthread_set_qos_class_self_np(code, 0) };
    res == 0
}

#[cfg(not(target_os = "macos"))]
pub fn set_qos(_class: QosClass) -> bool {
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_set_qos() {
        assert!(set_qos(QosClass::UserInteractive));
        assert!(set_qos(QosClass::Utility));
    }
}
