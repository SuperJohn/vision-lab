// Vision Lab.app — owns the Vision Lab processes so macOS TCC (Screen Recording, Mic, Camera)
// attributes them to com.houghtonlabs.visionlab instead of whatever terminal/agent launched them.
import AppKit

let home = FileManager.default.homeDirectoryForCurrentUser.path
let repo = ProcessInfo.processInfo.environment["VLAB_REPO"] ?? "\(home)/vision-lab"   // path to this checkout
let uiURL = URL(string: "http://127.0.0.1:8765")!

struct Svc { let name: String; let dir: String; let cmd: String }
let services = [
    Svc(name: "server", dir: repo, cmd: "exec ./run.sh >> data/server.log 2>&1"),
    Svc(name: "planloop", dir: repo, cmd: "exec .venv/bin/python -m vision_lab.planloop run >> data/planloop.log 2>&1"),
    Svc(name: "vlabbatch", dir: repo, cmd: "exec taskpolicy -b .venv/bin/python -m vision_lab.batch loop >> data/batch.log 2>&1"),
]

final class App: NSObject, NSApplicationDelegate {
    var procs: [String: Process] = [:]
    var quitting = false

    func start(_ s: Svc) {
        guard FileManager.default.fileExists(atPath: s.dir) else { return }
        let p = Process()
        p.executableURL = URL(fileURLWithPath: "/bin/zsh")
        p.arguments = ["-lc", s.cmd]
        p.currentDirectoryURL = URL(fileURLWithPath: s.dir)
        p.terminationHandler = { [weak self] _ in
            DispatchQueue.main.asyncAfter(deadline: .now() + 5) {   // keep services up: restart on crash
                guard let self = self, !self.quitting else { return }
                self.start(s)
            }
        }
        do { try p.run(); procs[s.name] = p } catch { NSLog("Vision Lab: failed to start \(s.name): \(error)") }
    }

    func applicationDidFinishLaunching(_ n: Notification) {
        let menu = NSMenu(); let item = NSMenuItem(); menu.addItem(item)
        let sub = NSMenu()
        sub.addItem(withTitle: "Open Vision Lab", action: #selector(openUI), keyEquivalent: "o")
        sub.addItem(.separator())
        sub.addItem(withTitle: "Quit Vision Lab", action: #selector(NSApplication.terminate(_:)), keyEquivalent: "q")
        item.submenu = sub; NSApp.mainMenu = menu
        services.forEach(start)
        DispatchQueue.main.asyncAfter(deadline: .now() + 6) { self.openUI() }
    }

    @objc func openUI() { NSWorkspace.shared.open(uiURL) }

    func applicationShouldHandleReopen(_ s: NSApplication, hasVisibleWindows f: Bool) -> Bool { openUI(); return true }

    func applicationWillTerminate(_ n: Notification) {
        quitting = true
        for p in procs.values where p.isRunning { kill(p.processIdentifier, SIGTERM) }
        // zsh exec's straight into the service, so SIGTERM reaches python/uvicorn directly
        let deadline = Date().addingTimeInterval(8)
        while procs.values.contains(where: { $0.isRunning }) && Date() < deadline { usleep(200_000) }
        for p in procs.values where p.isRunning { kill(p.processIdentifier, SIGKILL) }
    }
}

let app = NSApplication.shared
let delegate = App()
app.delegate = delegate
app.setActivationPolicy(.regular)
app.run()
