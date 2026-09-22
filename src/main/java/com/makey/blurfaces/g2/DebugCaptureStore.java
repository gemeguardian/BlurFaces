package com.makey.blurfaces.g2;

import java.io.IOException;
import java.nio.file.DirectoryStream;
import java.nio.file.Files;
import java.nio.file.LinkOption;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.nio.file.StandardOpenOption;
import java.util.UUID;

/** Local-only bounded store. Session consent is never persisted. No Android dependencies. */
final class DebugCaptureStore {
    static final long SESSION_NS = 300_000_000_000L;
    static final long INTERVAL_NS = 500_000_000L;
    static final long RETENTION_MS = 86_400_000L;
    static final int MAX_FILES = 300;
    static final long MAX_BYTES = 32L * 1024 * 1024;
    static final int MAX_RECORD_BYTES = 4 * 1024 * 1024;
    private final Path directory;
    private boolean enabled, busy;
    private long generation = 1, expiresNanos, nextSampleNanos;
    private String reason = "off";
    private int saved;
    private long dropped;

    DebugCaptureStore(Path directory) { this.directory = directory; }

    synchronized void configure(boolean value, long now, long wallMillis) throws IOException {
        enabled = false;
        generation++;
        reason = "off";
        if (Files.exists(directory, LinkOption.NOFOLLOW_LINKS)) prune(wallMillis);
        if (value) {
            ensureDirectory();
            expiresNanos = now + SESSION_NS;
            nextSampleNanos = now;
            saved = 0;
            dropped = 0;
            enabled = true;
            reason = "recording";
        }
    }

    synchronized boolean enabled(long now) {
        if (enabled && now >= expiresNanos) {
            enabled = false;
            generation++;
            reason = "session_expired";
        }
        return enabled;
    }

    synchronized long reserve(long now) {
        if (!enabled(now) || now < nextSampleNanos) return 0;
        if (busy) { dropped++; return 0; }
        nextSampleNanos = now + INTERVAL_NS;
        busy = true;
        return generation;
    }

    synchronized void release() { busy = false; }

    synchronized boolean accepts(long token, long now) {
        return token != 0 && enabled(now) && token == generation;
    }

    synchronized boolean write(long token, long now, byte[] zip) throws IOException {
        if (!accepts(token, now)) return false;
        if (zip.length > MAX_RECORD_BYTES) throw new IOException("Diagnostic sample exceeds size limit");
        ensureDirectory();
        long[] usage = usage();
        if (usage[0] >= MAX_FILES || usage[1] + zip.length > MAX_BYTES) {
            enabled = false;
            generation++;
            reason = "storage_limit";
            return false;
        }
        String name = "sample_" + UUID.randomUUID();
        Path temporary = directory.resolve(name + ".tmp");
        try {
            Files.write(temporary, zip, StandardOpenOption.CREATE_NEW, StandardOpenOption.WRITE);
            Files.move(temporary, directory.resolve(name + ".zip"), StandardCopyOption.ATOMIC_MOVE);
            saved++;
            return true;
        } finally {
            Files.deleteIfExists(temporary);
        }
    }

    synchronized void failure(long token) {
        if (token == generation) failure();
    }

    synchronized void failure() {
        enabled = false;
        generation++;
        reason = "io_or_capture_error";
    }

    synchronized String status(long now) throws IOException {
        enabled(now);
        long[] usage = usage();
        return "state=" + reason + ", files=" + usage[0] + ", bytes=" + usage[1]
                + ", saved=" + saved + ", busyDrops=" + dropped
                + ", remainingSec=" + (enabled ? Math.max(0L, (expiresNanos-now)/1_000_000_000L) : 0)
                + ", path=" + directory;
    }

    synchronized int clear() throws IOException {
        enabled = false;
        generation++;
        reason = "off";
        int count = 0;
        if (!Files.exists(directory, LinkOption.NOFOLLOW_LINKS)) return 0;
        checkDirectory();
        try (DirectoryStream<Path> files = Files.newDirectoryStream(directory)) {
            for (Path path : files) if (owned(path)) {
                Files.delete(path);
                count++;
            }
        }
        return count;
    }

    private void ensureDirectory() throws IOException {
        Files.createDirectories(directory);
        checkDirectory();
    }

    private void checkDirectory() throws IOException {
        if (Files.isSymbolicLink(directory) || !Files.isDirectory(directory, LinkOption.NOFOLLOW_LINKS))
            throw new IOException("Unsafe diagnostic directory");
    }

    private boolean owned(Path path) {
        return path.getFileName().toString().matches("sample_[0-9a-f-]{36}\\.(zip|tmp)")
                && Files.isRegularFile(path, LinkOption.NOFOLLOW_LINKS);
    }

    private long[] usage() throws IOException {
        long count = 0, bytes = 0;
        if (Files.exists(directory, LinkOption.NOFOLLOW_LINKS)) {
            checkDirectory();
            try (DirectoryStream<Path> files = Files.newDirectoryStream(directory)) {
                for (Path path : files) if (owned(path)) { count++; bytes += Files.size(path); }
            }
        }
        return new long[]{count, bytes};
    }

    private void prune(long wallMillis) throws IOException {
        checkDirectory();
        try (DirectoryStream<Path> files = Files.newDirectoryStream(directory)) {
            for (Path path : files) if (owned(path)
                    && wallMillis - Files.getLastModifiedTime(path, LinkOption.NOFOLLOW_LINKS).toMillis() > RETENTION_MS)
                Files.delete(path);
        }
    }
}
