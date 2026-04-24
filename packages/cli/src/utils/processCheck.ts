import { existsSync, readFileSync, writeFileSync } from 'fs';
import { PID_FILE, REFERENCE_COUNT_FILE } from '@CCR/shared';
import { readConfigFile } from '.';
import find from 'find-process';
import { execSync } from 'child_process'; // Import execSync to execute command line

export async function isProcessRunning(pid: number): Promise<boolean> {
    try {
        const processes = await find('pid', pid);
        return processes.length > 0;
    } catch (error) {
        return false;
    }
}

export function incrementReferenceCount() {
    let count = 0;
    if (existsSync(REFERENCE_COUNT_FILE)) {
        count = parseInt(readFileSync(REFERENCE_COUNT_FILE, 'utf-8')) || 0;
    }
    count++;
    writeFileSync(REFERENCE_COUNT_FILE, count.toString());
}

export function decrementReferenceCount() {
    let count = 0;
    if (existsSync(REFERENCE_COUNT_FILE)) {
        count = parseInt(readFileSync(REFERENCE_COUNT_FILE, 'utf-8')) || 0;
    }
    count = Math.max(0, count - 1);
    writeFileSync(REFERENCE_COUNT_FILE, count.toString());
}

export function getReferenceCount(): number {
    if (!existsSync(REFERENCE_COUNT_FILE)) {
        return 0;
    }
    return parseInt(readFileSync(REFERENCE_COUNT_FILE, 'utf-8')) || 0;
}

export function isServiceRunning(): boolean {
    if (!existsSync(PID_FILE)) {
        return false;
    }

    let pid: number;
    try {
        const pidStr = readFileSync(PID_FILE, 'utf-8');
        pid = parseInt(pidStr, 10);
        if (isNaN(pid)) {
            // PID file content is invalid
            cleanupPidFile();
            return false;
        }
    } catch (e) {
        // Read file failed
        return false;
    }

    try {
        if (process.platform === 'win32') {
            // --- Windows platform logic ---
            // Use tasklist command and find process via PID filter
            // stdio: 'pipe' suppresses command output to prevent it from showing in console
            const command = `tasklist /FI "PID eq ${pid}"`;
            const output = execSync(command, { stdio: 'pipe' }).toString();

            // If output contains PID, process exists
            // tasklist returns "INFO: No tasks are running..." when process not found
            // So a simple inclusion check is sufficient
            if (output.includes(pid.toString())) {
                return true;
            } else {
                // Theoretically if tasklist executes successfully but doesn't find, this won't be hit
                // But as a safeguard, we still consider process non-existent
                cleanupPidFile();
                return false;
            }

        } else {
            // --- Linux, macOS and other platform logic ---
            // Use signal 0 to check if process exists; this won't actually kill the process
            process.kill(pid, 0);
            return true; // If no exception thrown, process exists
        }
    } catch (e) {
        // Exception caught, process does not exist (whether kill or execSync failed)
        // Clean up invalid PID file
        cleanupPidFile();
        return false;
    }
}

export function savePid(pid: number) {
    writeFileSync(PID_FILE, pid.toString());
}

export function cleanupPidFile() {
    if (existsSync(PID_FILE)) {
        try {
            const fs = require('fs');
            fs.unlinkSync(PID_FILE);
        } catch (e) {
            // Ignore cleanup errors
        }
    }
}

export function getServicePid(): number | null {
    if (!existsSync(PID_FILE)) {
        return null;
    }

    try {
        const pid = parseInt(readFileSync(PID_FILE, 'utf-8'));
        return isNaN(pid) ? null : pid;
    } catch (e) {
        return null;
    }
}

export async function getServiceInfo() {
    const pid = getServicePid();
    const running = await isServiceRunning();
    const config = await readConfigFile();
    const port = config.PORT || 3456;

    return {
        running,
        pid,
        port,
        endpoint: `http://127.0.0.1:${port}`,
        pidFile: PID_FILE,
        referenceCount: getReferenceCount()
    };
}

export async function closeService() {
    // Check reference count
    const referenceCount = getReferenceCount();

    // Only stop the service if reference count is 0
    if (referenceCount === 0) {
        const pid = getServicePid();
        if (pid && await isServiceRunning()) {
            try {
                // Kill the service process
                process.kill(pid, 'SIGTERM');
            } catch (e) {
                // Ignore kill errors
            }
        }
    }
}
