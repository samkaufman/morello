#[cfg(test)]
static CELL: std::sync::OnceLock<std::sync::mpsc::Sender<()>> = std::sync::OnceLock::new();

#[cfg(test)]
#[ctor::ctor]
fn test_ctor_init() {
    use std::sync::mpsc;
    use std::{sync::mpsc::RecvTimeoutError, thread};

    CELL.get_or_init(|| {
        let (tx, rx) = mpsc::channel();
        let mut status = simple_server_status::SimpleServerStatus::default();

        thread::Builder::new()
            .name(String::from("SystemStatusThread"))
            .spawn(move || loop {
                match rx.recv_timeout(std::time::Duration::from_secs(1)) {
                    Ok(_) => return,
                    Err(RecvTimeoutError::Timeout) => {
                        match status.update() {
                            Ok(_) => println!("RAM used: {:?}", status.ram_usage()),
                            Err(_) => println!("couldn't measure RAM"),
                        };
                    }
                    Err(RecvTimeoutError::Disconnected) => {
                        panic!("Unexpectedly disconnected thread channel.");
                    }
                }
            })
            .unwrap();

        tx
    });
}

#[cfg(test)]
#[ctor::dtor]
fn test_ctor_drop() {
    let tx = CELL.get().unwrap();
    tx.send(()).unwrap();
}
