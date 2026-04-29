# Bastion Mobile — local debug APK

This describes how to produce a **standalone debug APK** on a Linux machine (no Android Studio, no Metro on the device). The same shape is used in GitHub Actions (`.github/workflows/build-mobile-android.yml`).

## Hard-coding the Bastion server URL

The app resolves the API origin in this order (see `src/api/config.ts`):

1. **Runtime** — value saved in SecureStore from the in-app **Server** screen (wins over build-time values).
2. **`process.env.EXPO_PUBLIC_API_BASE_URL`** — inlined when the **JavaScript bundle** is produced.
3. **`app.config.ts` → `extra.apiBaseUrl`** — baked into the native config at **prebuild** time.

To **hard-code a default URL into the APK** (so users land on your server without typing it), use **one or both** of:

### Option A — Environment variable at bundle time (recommended)

Create `mobile/.env` (do not commit; copy from `env.example`):

```bash
# mobile/.env — no trailing slash
EXPO_PUBLIC_API_BASE_URL=https://bastion.example.com
```

Expo loads `.env` when you run `npx expo prebuild` and when Gradle runs the Metro bundle step. Keep the same file in place for the whole build.

Alternatively, export in the shell before **both** prebuild and assemble:

```bash
export EXPO_PUBLIC_API_BASE_URL=https://bastion.example.com
```

### Option B — Literal fallback in `app.config.ts`

If you want a repo-visible default (e.g. internal CI host), set a non-empty fallback in [`app.config.ts`](app.config.ts):

```ts
extra: {
  apiBaseUrl:
    process.env.EXPO_PUBLIC_API_BASE_URL?.trim() ||
    'https://bastion.internal.example.com',
},
```

Rebuild with `npx expo prebuild --platform android` so native projects pick up `extra`.

### Option C — CI / one-off

In GitHub Actions, set `EXPO_PUBLIC_API_BASE_URL` as an env var on the **Expo prebuild** and **Gradle assemble** steps (same as local export).

**Note:** If a user later sets a URL in the app, that **runtime** value overrides build-time defaults until they clear it on the Server / Profile flow.

---

## Prerequisites (Debian / Ubuntu CLI)

| Requirement | Notes |
|-------------|--------|
| **Node.js 20** | e.g. NodeSource `setup_20.x` |
| **JDK 17** | `openjdk-17-jdk`; set `JAVA_HOME` (e.g. `/usr/lib/jvm/java-17-openjdk-amd64`) |
| **Android SDK** | Command-line tools only; install packages matching CI (below) |

### Android SDK packages (match CI)

From `sdkmanager` (after installing [command-line tools](https://developer.android.com/studio#command-line-tools-only)):

```bash
export ANDROID_SDK_ROOT="$HOME/android-sdk"
export PATH="$PATH:$ANDROID_SDK_ROOT/cmdline-tools/latest/bin:$ANDROID_SDK_ROOT/platform-tools"

yes | sdkmanager --licenses
sdkmanager "platform-tools" "platforms;android-35" "build-tools;35.0.0" "ndk;27.1.12297006"
```

Tell Gradle where the SDK is — either:

```bash
export ANDROID_HOME="$ANDROID_SDK_ROOT"
```

or create `mobile/android/local.properties` after the first prebuild:

```properties
sdk.dir=/home/YOU/android-sdk
```

(use your actual path; forward slashes on Linux)

---

## Build steps (from repo root)

### One-shot script (same steps)

From `mobile/`:

```bash
chmod +x build-debug-apk.sh
export ANDROID_SDK_ROOT="$HOME/android-sdk"   # or ANDROID_HOME
export JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
./build-debug-apk.sh
```

The script runs `npm install`, `expo prebuild --platform android --no-install`, ensures `android/local.properties` has `sdk.dir` when missing, then `NODE_ENV=production ./gradlew assembleDebug`. It loads `mobile/.env` if present (for `EXPO_PUBLIC_API_BASE_URL`).

### Manual commands (equivalent)

```bash
cd mobile
npm install
```

**1. Optional — fix API URL for this build**

- Ensure `mobile/.env` contains `EXPO_PUBLIC_API_BASE_URL=...`, or export it as above.

**2. Generate the Android project**

```bash
npx expo prebuild --platform android --no-install
```

Run again whenever you change `app.config.ts`, native plugins, or native dependencies.

**3. Assemble the debug APK** (embedded JS bundle; no Metro on device)

```bash
cd android
chmod +x gradlew
NODE_ENV=production ./gradlew assembleDebug --no-daemon
```

**4. Output artifact**

```text
mobile/android/app/build/outputs/apk/debug/*.apk
```

Copy that file to a device or emulator (`adb install …`) or to another machine.

---

## Troubleshooting

| Issue | What to check |
|-------|----------------|
| `JAVA_HOME` / no `java` | Install `openjdk-17-jdk`, export `JAVA_HOME` and `PATH`. |
| SDK location not found | `local.properties` `sdk.dir` or `ANDROID_HOME` / `ANDROID_SDK_ROOT`. |
| `Unable to load script` / missing bundle | Debug variant must embed the bundle: the repo uses a config plugin (`plugins/withAndroidEmbeddedDebugBundle.js`) and `NODE_ENV=production` for the Gradle bundle task. |
| Wrong or empty API host in APK | `EXPO_PUBLIC_API_BASE_URL` must be set **when Metro bundles** (Gradle `assembleDebug` with `NODE_ENV=production`). Re-run assemble after changing `.env`. |
| 16 KB / ELF warnings on API 35 | NDK `27.1.12297006` is set via `expo-build-properties` in `app.config.ts`; prebuilt RN 0.76 libs may still warn until RN/Expo upgrades. |

---

## Parity with GitHub Actions

The workflow runs: `npm install` → `npx expo prebuild --platform android --no-install` → `setup-java` 17 → `setup-android` with the same `sdkmanager` packages → `NODE_ENV=production ./gradlew assembleDebug` in `mobile/android`.
