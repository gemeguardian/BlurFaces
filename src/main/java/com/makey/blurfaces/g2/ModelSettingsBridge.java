package com.makey.blurfaces.g2;

import android.animation.ValueAnimator;
import android.content.Context;
import android.graphics.Canvas;
import android.graphics.Paint;
import android.graphics.Path;
import android.graphics.PorterDuff;
import android.graphics.RectF;
import android.graphics.drawable.ColorDrawable;
import android.graphics.drawable.Drawable;
import android.util.TypedValue;
import android.view.Gravity;
import android.view.View;
import android.view.ViewParent;
import android.view.accessibility.AccessibilityNodeInfo;
import android.widget.FrameLayout;
import android.widget.ImageView;
import android.widget.TextView;

import com.chaquo.python.PyObject;
import com.exteragram.messenger.plugins.Plugin;
import com.exteragram.messenger.plugins.models.CustomSetting;

import org.telegram.messenger.AndroidUtilities;
import org.telegram.messenger.LocaleController;
import org.telegram.messenger.R;
import org.telegram.ui.ActionBar.BaseFragment;
import org.telegram.ui.ActionBar.Theme;
import org.telegram.ui.Components.ItemOptions;
import org.telegram.ui.Components.CubicBezierInterpolator;
import org.telegram.ui.Components.RadioButton;
import org.telegram.ui.Components.RecyclerListView;
import org.telegram.ui.Components.UItem;
import org.telegram.ui.Components.UniversalAdapter;
import org.telegram.ui.Components.UniversalRecyclerView;
import org.telegram.ui.LaunchActivity;

import java.lang.ref.WeakReference;
import java.util.ArrayList;
import java.util.Locale;
import java.util.function.IntConsumer;
import java.util.concurrent.atomic.AtomicBoolean;

public final class ModelSettingsBridge {
    private static final ModelFactory[] FACTORIES = {
            new PreciseFactory(), new NearFactory(), new FarFactory(),
            new FlagshipPresetFactory(), new BalancedPresetFactory(),
            new EcoPresetFactory(), new ManualPresetFactory()
    };
    public static final int MODEL_COUNT = 3;
    public static final int PRESET_COUNT = 4;
    public static final int PRESET_FLAGSHIP = 3;
    public static final int PRESET_BALANCED = 4;
    public static final int PRESET_ECO = 5;
    public static final int PRESET_MANUAL = 6;

    private static volatile IntConsumer clickCallback;
    private static volatile IntConsumer deleteCallback;
    private static volatile IntConsumer presetCallback;
    private static volatile PresetSnapshot presetSnapshot = new PresetSnapshot(
            new String[] {"", "", "", ""}, "",
            new String[] {"", "", "", "", ""}, -1);
    private static final DownloadState presetDownload = new DownloadState(
            0, 0, 0L, -1L, -1);
    private static final ArrayList<WeakReference<ModelRadioCell>> cells = new ArrayList<>();
    private static volatile Snapshot snapshot = Snapshot.empty();
    private static final DownloadState[] downloadStates = {
            DownloadState.idle(), DownloadState.idle(), DownloadState.idle()
    };
    private static final Object DOWNLOAD_LOCK = new Object();
    private static final AtomicBoolean REFRESH_POSTED = new AtomicBoolean();

    public static final int DOWNLOAD_IDLE = 0;
    public static final int DOWNLOAD_STARTING = 1;
    public static final int DOWNLOAD_ACTIVE = 2;
    public static final int DOWNLOAD_VERIFYING = 3;
    public static final int DOWNLOAD_SUCCESS = 4;
    public static final int DOWNLOAD_FAILED = 5;

    static {
        for (ModelFactory factory : FACTORIES) {
            UItem.UItemFactory.setup(factory);
        }
    }

    private ModelSettingsBridge() { }

    public static CustomSetting.Factory<ModelRadioCell> getFactory(int index) {
        if (index < 0 || index >= MODEL_COUNT) throw new IllegalArgumentException("model index");
        return FACTORIES[index];
    }

    public static UItem getItem(int index) {
        if (index < 0 || index >= MODEL_COUNT) throw new IllegalArgumentException("model index");
        return UItem.ofFactory(FACTORIES[index].getClass());
    }

    public static synchronized void attach(IntConsumer newClickCallback,
                                            IntConsumer newDeleteCallback, String preciseLabel,
                                            String nearLabel, String farLabel, String deleteLabel,
                                            String connectingLabel, String downloadingLabel,
                                            String verifyingLabel, String downloadedLabel,
                                            String failedLabel,
                                            int selectedModel,
                                            int downloadedMask) {
        clickCallback = newClickCallback;
        deleteCallback = newDeleteCallback;
        snapshot = new Snapshot(
                 new String[] {safe(preciseLabel), safe(nearLabel), safe(farLabel)},
                 safe(deleteLabel), new String[] {safe(connectingLabel), safe(downloadingLabel),
                         safe(verifyingLabel), safe(downloadedLabel), safe(failedLabel)},
                 selectedModel, downloadedMask);
    }

    public static synchronized void attachPresets(IntConsumer newPresetCallback,
                                                  String flagshipLabel,
                                                  String balancedLabel, String ecoLabel,
                                                  String manualLabel,
                                                  String customSubtext,
                                                  String connectingLabel, String downloadingLabel,
                                                  String verifyingLabel, String downloadedLabel,
                                                  String failedLabel,
                                                  int selectedPreset) {
        presetCallback = newPresetCallback;
        presetSnapshot = new PresetSnapshot(
                new String[] {safe(flagshipLabel), safe(balancedLabel),
                        safe(ecoLabel), safe(manualLabel)},
                safe(customSubtext),
                new String[] {safe(connectingLabel), safe(downloadingLabel),
                        safe(verifyingLabel), safe(downloadedLabel), safe(failedLabel)},
                selectedPreset);
        refreshCells(false);
    }

    public static synchronized boolean updatePresets(IntConsumer expectedPresetCallback,
                                                     String flagshipLabel,
                                                     String balancedLabel, String ecoLabel,
                                                     String manualLabel,
                                                     String customSubtext,
                                                     String connectingLabel, String downloadingLabel,
                                                     String verifyingLabel, String downloadedLabel,
                                                     String failedLabel,
                                                     int selectedPreset) {
        if (presetCallback == null || presetCallback != expectedPresetCallback) return false;
        presetSnapshot = new PresetSnapshot(
                new String[] {safe(flagshipLabel), safe(balancedLabel),
                        safe(ecoLabel), safe(manualLabel)},
                safe(customSubtext),
                new String[] {safe(connectingLabel), safe(downloadingLabel),
                        safe(verifyingLabel), safe(downloadedLabel), safe(failedLabel)},
                selectedPreset);
        refreshCells(false);
        return true;
    }

    public static synchronized void detachPresets(IntConsumer expectedPresetCallback) {
        if (presetCallback == null || presetCallback != expectedPresetCallback) return;
        presetCallback = null;
        presetSnapshot = new PresetSnapshot(
                new String[] {"", "", "", ""}, "",
                new String[] {"", "", "", "", ""}, -1);
        synchronized (DOWNLOAD_LOCK) {
            presetDownload.phase = DOWNLOAD_IDLE;
        }
    }

    public static void updatePresetDownload(int index, int phase, long downloadedBytes,
                                            long totalBytes) {
        synchronized (DOWNLOAD_LOCK) {
            presetDownload.downloadingPreset = index;
            presetDownload.phase = phase;
            presetDownload.downloadedBytes = Math.max(0L, downloadedBytes);
            presetDownload.totalBytes = totalBytes;
        }
        refreshCells(true);
        if (phase == DOWNLOAD_FAILED || phase == DOWNLOAD_SUCCESS) {
            AndroidUtilities.runOnUIThread(() -> {
                synchronized (DOWNLOAD_LOCK) {
                    if (presetDownload.phase == phase) {
                        presetDownload.phase = DOWNLOAD_IDLE;
                    }
                }
                refreshCells(true);
            }, phase == DOWNLOAD_FAILED ? 1800 : 800);
        }
    }

    public static CustomSetting.Factory<ModelRadioCell> getPresetFactory(int index) {
        if (index < PRESET_FLAGSHIP || index > PRESET_MANUAL) {
            throw new IllegalArgumentException("preset index");
        }
        return FACTORIES[index];
    }

    public static UItem getPresetItem(int index) {
        if (index < PRESET_FLAGSHIP || index > PRESET_MANUAL) {
            throw new IllegalArgumentException("preset index");
        }
        return UItem.ofFactory(FACTORIES[index].getClass());
    }

    public static void onCellClick(int index, View view) {
        if (index < 0 || index >= FACTORIES.length) return;
        FACTORIES[index].onClick(null, null, view);
    }

    public static void onCellLongClick(int index, View view) {
        if (index < 0 || index >= FACTORIES.length) return;
        FACTORIES[index].onLongClick(null, null, view);
    }

    private static void log(String message) {
        try {
            com.exteragram.messenger.utils.AppUtils.log("[BlurFaces] " + message);
        } catch (Throwable ignored) {
        }
        android.util.Log.i("BlurFaces", message);
    }

    private static void log(String message, Throwable error) {
        try {
            com.exteragram.messenger.utils.AppUtils.log("[BlurFaces] " + message, error);
        } catch (Throwable ignored) {
        }
        android.util.Log.e("BlurFaces", message, error);
    }

    public static synchronized boolean update(IntConsumer expectedClickCallback,
                                               IntConsumer expectedDeleteCallback,
                                                 String preciseLabel,
                                                 String nearLabel, String farLabel, String deleteLabel,
                                                 String connectingLabel, String downloadingLabel,
                                                 String verifyingLabel, String downloadedLabel,
                                                 String failedLabel,
                                                 int selectedModel,
                                                 int downloadedMask) {
        if (!hasCallbacks(expectedClickCallback, expectedDeleteCallback)) return false;
        snapshot = new Snapshot(
                 new String[] {safe(preciseLabel), safe(nearLabel), safe(farLabel)},
                 safe(deleteLabel), new String[] {safe(connectingLabel), safe(downloadingLabel),
                         safe(verifyingLabel), safe(downloadedLabel), safe(failedLabel)},
                 selectedModel, downloadedMask);
        refreshCells(false);
        return true;
    }

    public static synchronized boolean detach(IntConsumer expectedClickCallback,
                                               IntConsumer expectedDeleteCallback) {
        if (!hasCallbacks(expectedClickCallback, expectedDeleteCallback)) return false;
        clickCallback = null;
        deleteCallback = null;
        snapshot = Snapshot.empty();
        for (int index = 0; index < downloadStates.length; index++) {
            downloadStates[index] = DownloadState.idle();
        }
        return true;
    }

    private static void dispatchPresetClick(int index) {
        IntConsumer target = presetCallback;
        log("Preset click dispatch index=" + index
                + " callback=" + (target == null ? "null" : System.identityHashCode(target)));
        if (target == null) {
            log("Preset click ignored: callback is detached");
            return;
        }
        try {
            target.accept(index);
            log("Preset click callback returned index=" + index);
        }
        catch (Throwable error) {
            log("Preset settings click callback failed", error);
        }
    }

    public static void updateDownload(int index, int phase, long downloadedBytes, long totalBytes) {
        if (index < 0 || index >= FACTORIES.length) return;
        synchronized (DOWNLOAD_LOCK) {
            downloadStates[index] = new DownloadState(index, phase,
                    Math.max(0L, downloadedBytes), totalBytes, -1);
        }
        refreshCells(true);
        if (phase == DOWNLOAD_FAILED) {
            AndroidUtilities.runOnUIThread(() -> {
                DownloadState state;
                synchronized (DOWNLOAD_LOCK) { state = downloadStates[index]; }
                if (state.phase == DOWNLOAD_FAILED) {
                    synchronized (DOWNLOAD_LOCK) { downloadStates[index] = DownloadState.idle(); }
                    refreshCells(true);
                }
            }, 1800);
        } else if (phase == DOWNLOAD_SUCCESS) {
            AndroidUtilities.runOnUIThread(() -> {
                DownloadState state;
                synchronized (DOWNLOAD_LOCK) { state = downloadStates[index]; }
                if (state.phase == DOWNLOAD_SUCCESS) {
                    synchronized (DOWNLOAD_LOCK) { downloadStates[index] = DownloadState.idle(); }
                    refreshCells(true);
                }
            }, 800);
        }
    }

    private static synchronized boolean beginDownload(int index) {
        synchronized (DOWNLOAD_LOCK) {
            DownloadState state = downloadStates[index];
            if (state.phase != DOWNLOAD_IDLE && state.phase != DOWNLOAD_FAILED) return false;
            downloadStates[index] = new DownloadState(index, DOWNLOAD_STARTING, 0L, -1L, -1);
        }
        refreshCells(true);
        return true;
    }

    private static void refreshCells(boolean animated) {
        if (!REFRESH_POSTED.compareAndSet(false, true)) return;
        AndroidUtilities.runOnUIThread(() -> {
            try {
                synchronized (cells) {
                    for (int i = cells.size() - 1; i >= 0; i--) {
                        ModelRadioCell cell = cells.get(i).get();
                        if (cell == null) cells.remove(i);
                        else cell.applySnapshot(animated);
                    }
                }
            } finally { REFRESH_POSTED.set(false); }
        });
    }

    private static boolean hasCallbacks(IntConsumer expectedClickCallback,
                                        IntConsumer expectedDeleteCallback) {
        return expectedClickCallback != null && expectedDeleteCallback != null
                && clickCallback == expectedClickCallback
                && deleteCallback == expectedDeleteCallback;
    }

    private static String safe(String value) {
        return value == null ? "" : value;
    }

    private static void dispatchClick(int index) {
        IntConsumer target = clickCallback;
        log("Model click dispatch index=" + index
                + " callback=" + (target == null ? "null" : System.identityHashCode(target)));
        if (target == null) return;
        try {
            target.accept(index);
            log("Model click callback returned index=" + index);
        }
        catch (Throwable error) {
            log("Model settings click callback failed", error);
        }
    }

    private static void dispatchDelete(int index) {
        IntConsumer target = deleteCallback;
        if (target == null) return;
        try { target.accept(index); }
        catch (Throwable error) {
            android.util.Log.e("BlurFaces", "Model deletion callback failed", error);
        }
    }

    private static void showDeleteOptions(int index, View row) {
        Snapshot state = snapshot;
        if ((state.downloadedMask & (1 << index)) == 0) return;
        BaseFragment fragment = LaunchActivity.getSafeLastFragment();
        if (fragment == null || fragment.getContext() == null) return;
        RecyclerListView recyclerListView = null;
        ViewParent parent = row.getParent();
        while (parent != null && !(parent instanceof RecyclerListView)) {
            parent = parent.getParent();
        }
        if (parent instanceof RecyclerListView) {
            recyclerListView = (RecyclerListView) parent;
        }
        Drawable scrimBackground = recyclerListView == null
                ? null : recyclerListView.getClipBackground(row);
        if (scrimBackground == null) {
            scrimBackground = new ColorDrawable(Theme.getColor(
                    Theme.key_windowBackgroundWhite, fragment.getResourceProvider()));
        }
        String deleteLabel = state.deleteLabel.isEmpty()
                ? LocaleController.getString(R.string.Delete) : state.deleteLabel;
        Drawable savedSelector = row.getBackground();
        row.setPressed(false);
        row.jumpDrawablesToCurrentState();
        row.setBackground(null);
        AtomicBoolean selectorRestored = new AtomicBoolean();
        Runnable restoreSelector = () -> {
            if (!selectorRestored.compareAndSet(false, true)) return;
            if (row.getBackground() == null) {
                row.setBackground(savedSelector);
                row.refreshDrawableState();
                row.invalidate();
            }
        };
        ItemOptions options = ItemOptions.makeOptions(fragment, row)
                .add(R.drawable.msg_delete, deleteLabel, true,
                        () -> dispatchDelete(index))
                .setBlur(true)
                .setDrawScrim(true)
                .setScrimViewBackground(scrimBackground)
                .forceBelowScrim(true)
                .forceBottom(false)
                .setGravity(Gravity.LEFT)
                .setOnDismiss(restoreSelector);
        View deleteButton = options.getLastView();
        if (deleteButton != null) {
            int buttonWidth = Math.max(AndroidUtilities.dp(1),
                    row.getWidth() - AndroidUtilities.dp(16));
            android.view.ViewGroup.LayoutParams params = deleteButton.getLayoutParams();
            if (params != null) {
                params.width = buttonWidth;
                deleteButton.setLayoutParams(params);
            }
            deleteButton.setMinimumWidth(buttonWidth);
        }
        options.show();
        if (!options.isShown()) restoreSelector.run();
    }

    private static final class Snapshot {
        final String[] labels;
        final String deleteLabel;
        final String[] statusLabels;
        final int selectedModel;
        final int downloadedMask;

        Snapshot(String[] labels, String deleteLabel, String[] statusLabels,
                 int selectedModel, int downloadedMask) {
            this.labels = labels;
            this.deleteLabel = deleteLabel;
            this.statusLabels = statusLabels;
            this.selectedModel = selectedModel;
            this.downloadedMask = downloadedMask;
        }

        static Snapshot empty() {
            return new Snapshot(new String[] {"", "", ""}, "",
                    new String[] {"", "", "", "", ""}, 0, 0);
        }
    }

    static final class PresetSnapshot {
        final String[] labels;
        final String customSubtext;
        final String[] statusLabels;
        final int selectedPreset;
        int downloadIndex = -1;
        int downloadPhase = DOWNLOAD_IDLE;
        long downloadBytes;
        long downloadTotal;

        PresetSnapshot(String[] labels, String customSubtext, String[] statusLabels,
                       int selectedPreset) {
            this.labels = labels;
            this.customSubtext = customSubtext;
            this.statusLabels = statusLabels;
            this.selectedPreset = selectedPreset;
        }
    }

    private static final class DownloadState {
        int index;
        int phase;
        long downloadedBytes;
        long totalBytes;
        int downloadingPreset;

        DownloadState(int index, int phase, long downloadedBytes, long totalBytes,
                      int downloadingPreset) {
            this.index = index;
            this.phase = phase;
            this.downloadedBytes = downloadedBytes;
            this.totalBytes = totalBytes;
            this.downloadingPreset = downloadingPreset;
        }

        static DownloadState idle() { return new DownloadState(-1, DOWNLOAD_IDLE, 0L, -1L, -1); }
    }

    public static final class ModelRadioCell extends FrameLayout {
        private final TextView textView;
        private final RadioButton radioButton;
        private final ImageView downloadIcon;
        private final TextView statusView;
        private final TextView amountView;
        private final ExpressiveProgressView progressView;
        private final Theme.ResourcesProvider resourcesProvider;
        private boolean needDivider;
        private int boundIndex = -1;
        private boolean presetBound;
        private float expansion;
        private ValueAnimator expansionAnimator;

        ModelRadioCell(Context context, Theme.ResourcesProvider resourcesProvider) {
            super(context);
            this.resourcesProvider = resourcesProvider;
            setMinimumHeight(AndroidUtilities.dp(50));
            setWillNotDraw(false);

            radioButton = new RadioButton(context);
            radioButton.setSize(AndroidUtilities.dp(20));
            radioButton.setClickable(false);
            addView(radioButton, new FrameLayout.LayoutParams(
                    AndroidUtilities.dp(22), AndroidUtilities.dp(22)));

            downloadIcon = new ImageView(context);
            downloadIcon.setImageResource(R.drawable.msg_download);
            downloadIcon.setScaleType(ImageView.ScaleType.CENTER);
            downloadIcon.setClickable(false);
            addView(downloadIcon, new FrameLayout.LayoutParams(
                    AndroidUtilities.dp(24), AndroidUtilities.dp(24)));

            textView = new TextView(context);
            textView.setTextSize(TypedValue.COMPLEX_UNIT_DIP, 16);
            textView.setSingleLine(true);
            textView.setGravity(Gravity.CENTER_VERTICAL);
            textView.setClickable(false);
            addView(textView, new FrameLayout.LayoutParams(
                    LayoutParams.MATCH_PARENT, LayoutParams.MATCH_PARENT));

            statusView = new TextView(context);
            statusView.setTextSize(TypedValue.COMPLEX_UNIT_DIP, 13);
            statusView.setSingleLine(true);
            statusView.setAlpha(0f);
            addView(statusView);

            amountView = new TextView(context);
            amountView.setTextSize(TypedValue.COMPLEX_UNIT_DIP, 12);
            amountView.setSingleLine(true);
            amountView.setAlpha(0f);
            addView(amountView);

            progressView = new ExpressiveProgressView(context, resourcesProvider);
            progressView.setAlpha(0f);
            addView(progressView);
        }

        void bind(int index, boolean divider) {
            Snapshot state = snapshot;
            boundIndex = index;
            presetBound = false;
            boolean downloaded = (state.downloadedMask & (1 << index)) != 0;
            boolean checked = downloaded && state.selectedModel == index;
            int textColor = Theme.getColor(
                    Theme.key_windowBackgroundWhiteBlackText, resourcesProvider);
            int iconColor = Theme.getColor(
                    Theme.key_windowBackgroundWhiteGrayIcon, resourcesProvider);

            textView.setText(state.labels[index]);
            textView.setTextColor(textColor);
            textView.setEnabled(true);
            radioButton.setColor(
                    Theme.getColor(Theme.key_radioBackground, resourcesProvider),
                    Theme.getColor(Theme.key_radioBackgroundChecked, resourcesProvider));
            radioButton.setChecked(checked, false);
            radioButton.setVisibility(downloaded ? View.VISIBLE : View.GONE);
            radioButton.setEnabled(true);
            downloadIcon.setVisibility(downloaded ? View.GONE : View.VISIBLE);
            downloadIcon.setColorFilter(iconColor, PorterDuff.Mode.SRC_IN);
            downloadIcon.setEnabled(true);
            needDivider = divider;
            setEnabled(true);
            setAlpha(1.0f);
            setBackground(Theme.createSelectorDrawable(
                    Theme.getColor(Theme.key_listSelector, resourcesProvider),
                    2));
            setContentDescription(state.labels[index]);
            invalidate();
            applyDownloadState(false);
        }

        void bindPreset(int index, boolean divider) {
            PresetSnapshot state = presetSnapshot;
            boundIndex = index;
            presetBound = true;
            boolean selected = state.selectedPreset == index;
            int textColor = Theme.getColor(
                    Theme.key_windowBackgroundWhiteBlackText, resourcesProvider);

            DownloadState download;
            synchronized (DOWNLOAD_LOCK) { download = new DownloadState(
                    presetDownload.index, presetDownload.phase,
                    presetDownload.downloadedBytes, presetDownload.totalBytes,
                    presetDownload.downloadingPreset); }
            boolean activeDownload = download.phase != DOWNLOAD_IDLE;
            boolean myDownload = activeDownload && download.downloadingPreset == index;

            textView.setText(state.labels[index]);
            textView.setTextColor(textColor);
            textView.setEnabled(true);
            radioButton.setColor(
                    Theme.getColor(Theme.key_radioBackground, resourcesProvider),
                    Theme.getColor(Theme.key_radioBackgroundChecked, resourcesProvider));
            radioButton.setChecked(selected, false);
            radioButton.setVisibility(myDownload ? View.GONE : View.VISIBLE);
            downloadIcon.setVisibility(View.GONE);
            statusView.setText("");
            amountView.setText("");
            progressView.setState(DOWNLOAD_IDLE, 0L, -1L);
            if (myDownload) {
                int phase = download.phase;
                String status = "";
                if (phase == DOWNLOAD_STARTING) status = state.statusLabels[0];
                else if (phase == DOWNLOAD_ACTIVE) status = state.statusLabels[1];
                else if (phase == DOWNLOAD_VERIFYING) status = state.statusLabels[2];
                else if (phase == DOWNLOAD_SUCCESS) status = state.statusLabels[3];
                else if (phase == DOWNLOAD_FAILED) status = state.statusLabels[4];
                statusView.setText(status);
                boolean known = download.totalBytes > 0;
                String amount = "";
                if (download.downloadedBytes > 0) {
                    amount = formatBytes(download.downloadedBytes);
                    if (known) amount += " / " + formatBytes(download.totalBytes);
                }
                amountView.setText(amount);
                if (known && download.downloadedBytes > 0 && phase != DOWNLOAD_STARTING) {
                    int percent = (int) Math.min(100L,
                            download.downloadedBytes * 100L / download.totalBytes);
                    statusView.setText(status + "  ·  " + percent + "%");
                }
                int secondary = Theme.getColor(
                        Theme.key_windowBackgroundWhiteGrayText2, resourcesProvider);
                int failure = Theme.getColor(Theme.key_text_RedBold, resourcesProvider);
                statusView.setTextColor(phase == DOWNLOAD_FAILED ? failure : secondary);
                amountView.setTextColor(secondary);
                progressView.setState(phase, download.downloadedBytes, download.totalBytes);
            }
            float target = myDownload ? 1f : 0f;
            if (expansion != target) {
                if (target > expansion) setExpansion(target);
                else animateExpansion(target);
            }
            needDivider = divider;
            setEnabled(!myDownload);
            setAlpha(1.0f);
            setBackground(Theme.createSelectorDrawable(
                    Theme.getColor(Theme.key_listSelector, resourcesProvider),
                    2));
            String description = state.labels[index];
            if (myDownload && !statusView.getText().toString().isEmpty()) {
                description += ", " + statusView.getText();
            }
            setContentDescription(description);
            invalidate();
        }

        void applyDownloadState(boolean animated) {
            if (boundIndex < 0) return;
            DownloadState download;
            synchronized (DOWNLOAD_LOCK) { download = downloadStates[boundIndex]; }
            boolean active = download.phase != DOWNLOAD_IDLE;
            float target = active ? 1f : 0f;
            if (animated && expansion != target) animateExpansion(target);
            else if (!animated) setExpansion(target);

            Snapshot state = snapshot;
            int phase = active ? download.phase : DOWNLOAD_IDLE;
            String status = "";
            if (phase == DOWNLOAD_STARTING) status = state.statusLabels[0];
            else if (phase == DOWNLOAD_ACTIVE) status = state.statusLabels[1];
            else if (phase == DOWNLOAD_VERIFYING) status = state.statusLabels[2];
            else if (phase == DOWNLOAD_SUCCESS) status = state.statusLabels[3];
            else if (phase == DOWNLOAD_FAILED) status = state.statusLabels[4];
            statusView.setText(status);

            boolean known = download.totalBytes > 0;
            String amount = "";
            if (active && download.downloadedBytes > 0) {
                amount = formatBytes(download.downloadedBytes);
                if (known) amount += " / " + formatBytes(download.totalBytes);
            }
            amountView.setText(amount);
            if (active && known && phase != DOWNLOAD_STARTING) {
                int percent = (int) Math.min(100L,
                        download.downloadedBytes * 100L / download.totalBytes);
                statusView.setText(status + "  ·  " + percent + "%");
            }
            int secondary = Theme.getColor(Theme.key_windowBackgroundWhiteGrayText2, resourcesProvider);
            int failure = Theme.getColor(Theme.key_text_RedBold, resourcesProvider);
            statusView.setTextColor(phase == DOWNLOAD_FAILED ? failure : secondary);
            amountView.setTextColor(secondary);
            progressView.setState(phase, download.downloadedBytes, download.totalBytes);
            downloadIcon.setVisibility(active ? View.GONE
                    : (((state.downloadedMask & (1 << boundIndex)) != 0) ? View.GONE : View.VISIBLE));
            if (active) {
                radioButton.setVisibility(View.GONE);
            } else {
                boolean downloaded = (state.downloadedMask & (1 << boundIndex)) != 0;
                radioButton.setVisibility(downloaded ? View.VISIBLE : View.GONE);
                radioButton.setChecked(downloaded && state.selectedModel == boundIndex, animated);
            }
            setContentDescription(state.labels[boundIndex] + (status.isEmpty() ? "" : ", " + status)
                    + (amount.isEmpty() ? "" : ", " + amount));
        }

        void applySnapshot(boolean animated) {
            if (boundIndex < 0) return;
            if (presetBound) {
                bindPreset(boundIndex, needDivider);
                sendAccessibilityEvent(
                        android.view.accessibility.AccessibilityEvent.TYPE_WINDOW_CONTENT_CHANGED);
                invalidate();
                return;
            }
            Snapshot state = snapshot;
            textView.setText(state.labels[boundIndex]);
            applyDownloadState(animated);
            sendAccessibilityEvent(android.view.accessibility.AccessibilityEvent.TYPE_WINDOW_CONTENT_CHANGED);
            invalidate();
        }

        private void animateExpansion(float target) {
            if (expansionAnimator != null) expansionAnimator.cancel();
            expansionAnimator = ValueAnimator.ofFloat(expansion, target);
            expansionAnimator.setDuration(target > expansion ? 180 : 140);
            expansionAnimator.setInterpolator(target > expansion
                    ? CubicBezierInterpolator.Emphasized : CubicBezierInterpolator.EASE_OUT_QUINT);
            expansionAnimator.addUpdateListener(animation -> setExpansion((float) animation.getAnimatedValue()));
            expansionAnimator.start();
        }

        private void setExpansion(float value) {
            expansion = value;
            statusView.setAlpha(value);
            amountView.setAlpha(value);
            progressView.setAlpha(value);
            requestLayout();
            invalidate();
        }

        private static String formatBytes(long bytes) {
            if (bytes < 1024) return bytes + " B";
            if (bytes < 1024L * 1024L) return String.format(Locale.US, "%.1f KB", bytes / 1024f);
            return String.format(Locale.US, "%.1f MB", bytes / (1024f * 1024f));
        }

        @Override
        protected void onMeasure(int widthMeasureSpec, int heightMeasureSpec) {
            int width = MeasureSpec.getSize(widthMeasureSpec);
            int height = AndroidUtilities.dp(50) + Math.round(AndroidUtilities.dp(90) * expansion)
                    + (needDivider ? 1 : 0);
            int control = AndroidUtilities.dp(24);
            int textWidth = Math.max(0, width - AndroidUtilities.dp(84));
            radioButton.measure(MeasureSpec.makeMeasureSpec(AndroidUtilities.dp(22), MeasureSpec.EXACTLY),
                    MeasureSpec.makeMeasureSpec(AndroidUtilities.dp(22), MeasureSpec.EXACTLY));
            downloadIcon.measure(MeasureSpec.makeMeasureSpec(control, MeasureSpec.EXACTLY),
                    MeasureSpec.makeMeasureSpec(control, MeasureSpec.EXACTLY));
            textView.measure(MeasureSpec.makeMeasureSpec(textWidth, MeasureSpec.EXACTLY),
                    MeasureSpec.makeMeasureSpec(AndroidUtilities.dp(50), MeasureSpec.EXACTLY));
            int innerWidth = Math.max(0, width - AndroidUtilities.dp(56));
            statusView.measure(MeasureSpec.makeMeasureSpec(innerWidth, MeasureSpec.EXACTLY),
                    MeasureSpec.makeMeasureSpec(AndroidUtilities.dp(20), MeasureSpec.EXACTLY));
            amountView.measure(MeasureSpec.makeMeasureSpec(innerWidth, MeasureSpec.EXACTLY),
                    MeasureSpec.makeMeasureSpec(AndroidUtilities.dp(18), MeasureSpec.EXACTLY));
            progressView.measure(MeasureSpec.makeMeasureSpec(innerWidth, MeasureSpec.EXACTLY),
                    MeasureSpec.makeMeasureSpec(AndroidUtilities.dp(10), MeasureSpec.EXACTLY));
            setMeasuredDimension(width, height);
        }

        @Override
        protected void onLayout(boolean changed, int left, int top, int right, int bottom) {
            int width = right - left;
            int height = bottom - top;
            int radioSize = AndroidUtilities.dp(22);
            int iconSize = AndroidUtilities.dp(24);
            int controlCenter = width - AndroidUtilities.dp(31);
            int downloadCenter = controlCenter;
            int radioLeft = controlCenter - radioSize / 2;
            int iconLeft = downloadCenter - iconSize / 2;
            radioButton.layout(radioLeft, (height - radioSize) / 2,
                    radioLeft + radioSize, (height + radioSize) / 2);
            downloadIcon.layout(iconLeft, (height - iconSize) / 2,
                    iconLeft + iconSize, (height + iconSize) / 2);
            if (LocaleController.isRTL) {
                textView.setGravity(Gravity.RIGHT | Gravity.CENTER_VERTICAL);
                textView.layout(AndroidUtilities.dp(24), 0,
                        width - AndroidUtilities.dp(60), height);
            } else {
                textView.setGravity(Gravity.LEFT | Gravity.CENTER_VERTICAL);
                textView.layout(AndroidUtilities.dp(24), 0,
                        width - AndroidUtilities.dp(60), height);
            }
            int contentLeft = AndroidUtilities.dp(24);
            int contentRight = width - AndroidUtilities.dp(28);
            statusView.setGravity(LocaleController.isRTL ? Gravity.RIGHT : Gravity.LEFT);
            amountView.setGravity(LocaleController.isRTL ? Gravity.RIGHT : Gravity.LEFT);
            statusView.layout(contentLeft, AndroidUtilities.dp(41), contentRight, AndroidUtilities.dp(61));
            progressView.layout(contentLeft, AndroidUtilities.dp(88), contentRight, AndroidUtilities.dp(96));
            amountView.layout(contentLeft, AndroidUtilities.dp(114), contentRight, AndroidUtilities.dp(132));
        }

        @Override
        protected void onDraw(Canvas canvas) {
            super.onDraw(canvas);
            if (needDivider) {
                float start = AndroidUtilities.dp(24);
                float end = getWidth();
                canvas.drawLine(start, getHeight() - 1, end, getHeight() - 1, Theme.dividerPaint);
            }
        }

        @Override
        public void onInitializeAccessibilityNodeInfo(AccessibilityNodeInfo info) {
            super.onInitializeAccessibilityNodeInfo(info);
            info.setClassName("android.widget.RadioButton");
            info.setCheckable(radioButton.getVisibility() == View.VISIBLE);
            info.setChecked(radioButton.isChecked());
        }

        void cancelPendingClick() {
            ViewParent parent = getParent();
            while (parent != null && !(parent instanceof RecyclerListView)) {
                parent = parent.getParent();
            }
            if (parent instanceof RecyclerListView) {
                ((RecyclerListView) parent).cancelClickRunnables(true);
            }
        }

        @Override
        protected void onAttachedToWindow() {
            super.onAttachedToWindow();
            synchronized (cells) { cells.add(new WeakReference<>(this)); }
        }

        @Override
        protected void onDetachedFromWindow() {
            if (expansionAnimator != null) expansionAnimator.cancel();
            synchronized (cells) {
                for (int i = cells.size() - 1; i >= 0; i--) {
                    ModelRadioCell cell = cells.get(i).get();
                    if (cell == null || cell == this) cells.remove(i);
                }
            }
            super.onDetachedFromWindow();
        }
    }

    private static final class ExpressiveProgressView extends View {
        private final Paint paint = new Paint(Paint.ANTI_ALIAS_FLAG);
        private final Path wave = new Path();
        private final RectF track = new RectF();
        private final Theme.ResourcesProvider resourcesProvider;
        private int phase;
        private long downloaded;
        private long total;
        private long startedAt = System.currentTimeMillis();

        ExpressiveProgressView(Context context, Theme.ResourcesProvider resourcesProvider) {
            super(context);
            this.resourcesProvider = resourcesProvider;
        }

        void setState(int phase, long downloaded, long total) {
            if (this.phase != phase) startedAt = System.currentTimeMillis();
            this.phase = phase;
            this.downloaded = downloaded;
            this.total = total;
            invalidate();
        }

        @Override
        protected void onDraw(Canvas canvas) {
            super.onDraw(canvas);
            float center = getHeight() / 2f;
            float stroke = AndroidUtilities.dp(3);
            track.set(0, center - stroke / 2f, getWidth(), center + stroke / 2f);
            int secondary = Theme.getColor(Theme.key_windowBackgroundWhiteGrayText2, resourcesProvider);
            paint.setStyle(Paint.Style.FILL);
            paint.setColor((secondary & 0x00FFFFFF) | 0x28000000);
            canvas.drawRoundRect(track, stroke / 2f, stroke / 2f, paint);

            int color = Theme.getColor(phase == DOWNLOAD_FAILED ? Theme.key_text_RedBold
                    : Theme.key_windowBackgroundWhiteBlueText, resourcesProvider);
            paint.setColor(color);
            paint.setStyle(Paint.Style.STROKE);
            paint.setStrokeWidth(stroke);
            paint.setStrokeCap(Paint.Cap.ROUND);
            float progress = total > 0 ? Math.min(1f, downloaded / (float) total) : -1f;
            long elapsed = System.currentTimeMillis() - startedAt;
            float start;
            float end;
            if (progress >= 0f) {
                start = stroke / 2f;
                end = Math.max(start, (getWidth() - stroke) * progress + stroke / 2f);
            } else {
                float travel = (elapsed % 1200L) / 1200f;
                float segment = getWidth() * .28f;
                start = -segment + (getWidth() + segment) * travel;
                end = start + segment;
            }
            start = Math.max(stroke / 2f, start);
            end = Math.min(getWidth() - stroke / 2f, end);
            if (end > start) {
                wave.reset();
                float amplitude = phase == DOWNLOAD_ACTIVE ? AndroidUtilities.dp(1.25f) : 0f;
                float wavelength = AndroidUtilities.dp(45);
                float phaseOffset = (elapsed % 1000L) / 1000f * wavelength;
                for (float x = start; x <= end + 1f; x += AndroidUtilities.dpf2(1.5f)) {
                    float y = center + amplitude * (float) Math.sin(
                            ((x + phaseOffset) / wavelength) * Math.PI * 2.0);
                    if (x == start) wave.moveTo(x, y); else wave.lineTo(Math.min(x, end), y);
                }
                canvas.drawPath(wave, paint);
            }
            if (phase == DOWNLOAD_STARTING || phase == DOWNLOAD_ACTIVE || phase == DOWNLOAD_VERIFYING) {
                postInvalidateOnAnimation();
            }
        }
    }

    private abstract static class ModelFactory extends CustomSetting.Factory<ModelRadioCell> {
        final int index;

        ModelFactory(int index) {
            this.index = index;
            setClickableValue(true);
            setShadowValue(false);
        }

        boolean isPreset() { return false; }

        @Override
        public UItem create(Plugin plugin, CustomSetting setting, PyObject args) {
            return UItem.ofFactory(getClass());
        }

        @Override
        public ModelRadioCell createView(Context context, RecyclerListView listView,
                                         int currentAccount, int classGuid,
                                         Theme.ResourcesProvider resourcesProvider) {
            return new ModelRadioCell(context, resourcesProvider);
        }

        @Override
        public void bindView(View view, UItem item, boolean divider, UniversalAdapter adapter,
                             UniversalRecyclerView listView) {
            if (isPreset()) {
                ((ModelRadioCell) view).bindPreset(index - PRESET_FLAGSHIP, divider);
            } else {
                ((ModelRadioCell) view).bind(index, divider);
            }
        }

        @Override
        public void onClick(Plugin plugin, UItem item, View view) {
            if (isPreset()) {
                int localIndex = index - PRESET_FLAGSHIP;
                log("Preset cell onClick factory=" + index
                        + " local=" + localIndex + " selected=" + presetSnapshot.selectedPreset);
                PresetSnapshot current = presetSnapshot;
                if (current.selectedPreset != localIndex) {
                    presetSnapshot = new PresetSnapshot(
                            current.labels, current.customSubtext,
                            current.statusLabels, localIndex);
                    refreshCells(false);
                }
                dispatchPresetClick(index);
                return;
            }
            Snapshot state = snapshot;
            if ((state.downloadedMask & (1 << index)) == 0) {
                if (!beginDownload(index)) return;
            }
            dispatchClick(index);
            state = snapshot;
            if ((state.downloadedMask & (1 << index)) != 0) {
                if (state.selectedModel != index) {
                    snapshot = new Snapshot(state.labels, state.deleteLabel, state.statusLabels,
                            index, state.downloadedMask);
                }
                refreshCells(false);
            }
        }

        @Override
        public void onLongClick(Plugin plugin, UItem item, View view) {
            if (view instanceof ModelRadioCell) {
                ((ModelRadioCell) view).cancelPendingClick();
            }
            if (isPreset()) return;
            if (view != null) {
                showDeleteOptions(index, view);
            }
        }
    }

    public static final class PreciseFactory extends ModelFactory {
        PreciseFactory() { super(0); }
    }

    public static final class NearFactory extends ModelFactory {
        NearFactory() { super(1); }
    }

    public static final class FarFactory extends ModelFactory {
        FarFactory() { super(2); }
    }

    public static final class FlagshipPresetFactory extends ModelFactory {
        FlagshipPresetFactory() { super(PRESET_FLAGSHIP); }
        @Override public boolean isPreset() { return true; }
    }

    public static final class BalancedPresetFactory extends ModelFactory {
        BalancedPresetFactory() { super(PRESET_BALANCED); }
        @Override public boolean isPreset() { return true; }
    }

    public static final class EcoPresetFactory extends ModelFactory {
        EcoPresetFactory() { super(PRESET_ECO); }
        @Override public boolean isPreset() { return true; }
    }

    public static final class ManualPresetFactory extends ModelFactory {
        ManualPresetFactory() { super(PRESET_MANUAL); }
        @Override public boolean isPreset() { return true; }
    }
}
