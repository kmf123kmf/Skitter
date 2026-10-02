"""Step 2: divide the final source image into target regions for tile matching.

The user picks the mosaic layout (tile aspect and columns; pixel sizes are
chosen only on export) and builds a slicing plan: an ordered list of
operations (see skitter.core.slicing). The page lists the stages, generates a settings form
for the selected one from its declared parameters, and draws the resulting
regions over the image as they change. Regions may overlap: the preview
draws them in stacking order, so upper regions hide what they cover.
"""

import math

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QColor, QIcon, QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QSlider,
    QSpinBox,
    QStyle,
    QToolButton,
    QVBoxLayout,
)

from skitter.core.slicing import (
    TILE_ASPECTS,
    SlicingOperation,
    SlicingPlan,
    Stage,
    operation_types,
)
from skitter.ui import preferences
from skitter.ui.steps.base import StepPage, side_panel
from skitter.ui.widgets.image_viewer import ImageViewer
from skitter.ui.widgets.param_form import ParamForm
from skitter.ui.widgets.region_overlay import (
    DEFAULT_LINE_COLOR,
    LINE_COLORS,
    OUTLINES,
    STACKED,
    RegionOverlay,
)

UNCOVERED_DIMMING = 0.6
MIN_LINE_OPACITY, DEFAULT_LINE_OPACITY = 10, 100  # percent
MIN_SOURCE_PX_PER_TILE = 2.0  # below this, tile colors come from too few source pixels
WARNING_STYLE = "color: #c42b1c;"
HIDDEN = "hidden"
DISPLAY_MODES = (
    (STACKED, "Stacked", "Regions show the image; upper regions hide what they cover."),
    (OUTLINES, "Outlines", "Outlines only: every region's full extent is visible."),
    (HIDDEN, "Hidden", "Show the image without regions."),
)


def _stage_text(stage: Stage) -> str:
    summary = stage.operation.summary()
    return f"{stage.operation.name} — {summary}" if summary else stage.operation.name


def _muted(label: QLabel) -> QLabel:
    label.setWordWrap(True)
    label.setStyleSheet("color: palette(placeholder-text);")
    return label


class SlicingStep(StepPage):
    title = "Slicing"

    def __init__(self, session, parent=None):
        super().__init__(session, parent)
        self.viewer = ImageViewer()
        self.overlay = RegionOverlay(self.viewer.canvas)

        # Coalesce bursts of edits (e.g. holding a spin box arrow) into one run.
        self._recompute = QTimer(self)
        self._recompute.setSingleShot(True)
        self._recompute.setInterval(0)
        self._recompute.timeout.connect(session.slicing_edited)
        self._relayout = QTimer(self)
        self._relayout.setSingleShot(True)
        self._relayout.setInterval(0)
        self._relayout.timeout.connect(self._apply_layout)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self.viewer, stretch=1)
        layout.addWidget(
            side_panel(
                self._build_mosaic_group(),
                self._build_plan_group(),
                self._build_settings_group(),
                self._build_result_group(),
                self._build_display_group(),
            )
        )

        session.source_committed.connect(self._on_source_committed)
        session.layout_changed.connect(self._on_layout_changed)
        session.slicing_changed.connect(self._on_slicing_changed)
        self.viewer.canvas.cursor_moved.connect(self._on_hover)
        self.viewer.canvas.cursor_left.connect(lambda: self._on_hover(None, None))
        self._refresh_stages(select=0)
        self._apply_display()
        self._apply_line_style()
        self._show_layout()

    @property
    def plan(self) -> SlicingPlan:
        return self.session.project.slicing_plan

    # Construction

    def _build_mosaic_group(self) -> QGroupBox:
        self.tile_aspect = QComboBox()
        for value, label in TILE_ASPECTS:
            self.tile_aspect.addItem(label, value)
        self.tile_aspect.setToolTip("Shape of a base tile (width : height).")
        self.columns = QSpinBox()
        self.columns.setRange(1, 5000)
        self.columns.setToolTip("Base tiles across the mosaic. Rows follow the image's shape.")
        self.columns.setKeyboardTracking(False)
        self.columns.valueChanged.connect(lambda _: self._relayout.start())
        self.tile_aspect.currentIndexChanged.connect(lambda _: self._relayout.start())

        self._rows = QLabel("—")
        self._source_px = QLabel("—")
        self._source_px.setToolTip(
            "Source image pixels across one base tile: how much of the image "
            "each tile's color is judged from."
        )
        hint = _muted(QLabel("Pixel sizes are chosen when you export the mosaic."))

        group = QGroupBox("Mosaic")
        form = QFormLayout(group)
        form.addRow("Tile aspect:", self.tile_aspect)
        form.addRow("Columns:", self.columns)
        form.addRow("Rows:", self._rows)
        form.addRow("Source per tile:", self._source_px)
        form.addRow(hint)
        return group

    def _build_plan_group(self) -> QGroupBox:
        self._stages = QListWidget()
        self._stages.setMaximumHeight(140)
        self._stages.currentRowChanged.connect(self._on_stage_selected)
        self._stages.itemChanged.connect(self._on_stage_toggled)

        self.add_menu = QMenu(self)
        category = None
        for cls in operation_types():
            if cls.category != category:
                category = cls.category
                self.add_menu.addSection(category)
            action = self.add_menu.addAction(cls.name)
            action.setToolTip(cls.description)
            action.triggered.connect(lambda _=False, c=cls: self.add_stage(c()))
        self.add_menu.setToolTipsVisible(True)

        style = self.style()
        add_button = QToolButton()
        add_button.setText("Add")
        add_button.setMenu(self.add_menu)
        add_button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self._remove_button = self._tool_button(
            style.standardIcon(QStyle.StandardPixmap.SP_TrashIcon),
            "Remove stage",
            self.remove_stage,
        )
        self._up_button = self._tool_button(
            style.standardIcon(QStyle.StandardPixmap.SP_ArrowUp), "Move up",
            lambda: self.move_stage(-1),
        )  # fmt: skip
        self._down_button = self._tool_button(
            style.standardIcon(QStyle.StandardPixmap.SP_ArrowDown), "Move down",
            lambda: self.move_stage(1),
        )  # fmt: skip
        buttons = QHBoxLayout()
        buttons.addWidget(add_button)
        buttons.addStretch()
        for button in (self._up_button, self._down_button, self._remove_button):
            buttons.addWidget(button)

        hint = _muted(QLabel("Stages run top to bottom, starting from the whole mosaic."))
        group = QGroupBox("Slicing Plan")
        layout = QVBoxLayout(group)
        layout.addWidget(self._stages)
        layout.addLayout(buttons)
        layout.addWidget(hint)
        return group

    @staticmethod
    def _tool_button(icon, tooltip, slot) -> QToolButton:
        button = QToolButton()
        button.setIcon(icon)
        button.setToolTip(tooltip)
        button.clicked.connect(slot)
        return button

    def _build_settings_group(self) -> QGroupBox:
        self._description = _muted(QLabel())
        self.form = ParamForm()
        self.form.changed.connect(self._on_param_changed)
        self._settings_group = QGroupBox("Settings")
        layout = QVBoxLayout(self._settings_group)
        layout.addWidget(self._description)
        layout.addWidget(self.form)
        return self._settings_group

    def _build_result_group(self) -> QGroupBox:
        self._count = QLabel("—")
        self._density = QLabel("—")
        self._density.setToolTip(
            "Regions compared with a plain grid of base tiles covering the mosaic "
            "(a grid is about 1.0×)."
        )
        self._coverage = QLabel("—")
        self._coverage.setToolTip("Share of the mosaic covered by regions (estimated).")
        self._smallest = QLabel("—")
        self._median = QLabel("—")
        self._largest = QLabel("—")
        self._hover = QLabel("—")
        self._hover.setWordWrap(True)
        self._error = QLabel()
        self._error.setWordWrap(True)
        self._error.setStyleSheet(WARNING_STYLE)
        self._error.hide()

        group = QGroupBox("Result")
        form = QFormLayout(group)
        form.addRow("Regions:", self._count)
        form.addRow("Density:", self._density)
        form.addRow("Coverage:", self._coverage)
        form.addRow("Smallest:", self._smallest)
        form.addRow("Median:", self._median)
        form.addRow("Largest:", self._largest)
        form.addRow("Under cursor:", self._hover)
        form.addRow(self._error)
        return group

    def _build_display_group(self) -> QGroupBox:
        self.display_mode = QComboBox()
        for value, label, tip in DISPLAY_MODES:
            self.display_mode.addItem(label, value)
            index = self.display_mode.count() - 1
            self.display_mode.setItemData(index, tip, Qt.ItemDataRole.ToolTipRole)
        self.display_mode.currentIndexChanged.connect(self._apply_display)
        self.shadows = QCheckBox("Drop shadows")
        self.shadows.setToolTip("Shadow each region onto the regions below it.")
        self.shadows.toggled.connect(self._apply_display)
        self.dim_uncovered = QCheckBox("Dim uncovered areas")
        self.dim_uncovered.setToolTip("Darken parts of the image that no region covers.")
        self.dim_uncovered.setChecked(True)
        self.dim_uncovered.toggled.connect(self._apply_display)

        prefs = preferences.settings()
        self.line_color = QComboBox()
        for color_id, label, rgb in LINE_COLORS:
            swatch = QPixmap(14, 14)
            swatch.fill(QColor.fromRgbF(*rgb))
            self.line_color.addItem(QIcon(swatch), label, color_id)
        index = self.line_color.findData(prefs.value("slicing/overlay_color", DEFAULT_LINE_COLOR))
        self.line_color.setCurrentIndex(max(index, 0))
        self.line_color.setToolTip("Color of the region outlines.")
        self.line_opacity = QSlider(Qt.Orientation.Horizontal)
        self.line_opacity.setRange(MIN_LINE_OPACITY, 100)
        try:
            opacity = int(prefs.value("slicing/overlay_alpha", DEFAULT_LINE_OPACITY))
        except (TypeError, ValueError):
            opacity = DEFAULT_LINE_OPACITY
        self.line_opacity.setValue(min(max(opacity, MIN_LINE_OPACITY), 100))
        self.line_opacity.setToolTip("Opacity of the region outlines.")
        self._line_opacity_label = QLabel()
        self._line_opacity_label.setMinimumWidth(36)
        opacity_row = QHBoxLayout()
        opacity_row.addWidget(self.line_opacity, stretch=1)
        opacity_row.addWidget(self._line_opacity_label)
        self.line_color.currentIndexChanged.connect(self._apply_line_style)
        self.line_opacity.valueChanged.connect(self._apply_line_style)

        group = QGroupBox("Display")
        form = QFormLayout(group)
        form.addRow("Regions:", self.display_mode)
        form.addRow("Line color:", self.line_color)
        form.addRow("Line opacity:", opacity_row)
        form.addRow(self.shadows)
        form.addRow(self.dim_uncovered)
        return group

    def _apply_display(self) -> None:
        mode = self.display_mode.currentData()
        stacked = mode == STACKED
        self.overlay.set_visible(mode != HIDDEN)
        if mode != HIDDEN:
            self.overlay.set_mode(mode)
        self.overlay.set_shadows(stacked and self.shadows.isChecked())
        self.shadows.setEnabled(stacked)
        self.dim_uncovered.setEnabled(stacked)
        self.line_color.setEnabled(mode != HIDDEN)
        self.line_opacity.setEnabled(mode != HIDDEN)
        # Dimming only reads correctly when regions repaint the image they cover.
        dim = stacked and self.dim_uncovered.isChecked()
        self.viewer.set_dimming(UNCOVERED_DIMMING if dim else 0.0)

    def _apply_line_style(self) -> None:
        color_id = self.line_color.currentData()
        opacity = self.line_opacity.value()
        rgb = next(rgb for cid, _, rgb in LINE_COLORS if cid == color_id)
        self.overlay.set_line_color(rgb)
        self.overlay.set_line_alpha(opacity / 100)
        self._line_opacity_label.setText(f"{opacity}%")
        prefs = preferences.settings()
        prefs.setValue("slicing/overlay_color", color_id)
        prefs.setValue("slicing/overlay_alpha", opacity)

    # Mosaic layout

    def _show_layout(self) -> None:
        """Put the project's layout into the controls."""
        layout = self.session.project.layout
        for widget in (self.tile_aspect, self.columns):
            widget.blockSignals(True)
        index = min(
            range(self.tile_aspect.count()),
            key=lambda i: abs(self.tile_aspect.itemData(i) - layout.tile_aspect),
        )
        self.tile_aspect.setCurrentIndex(index)
        self.columns.setValue(layout.columns)
        for widget in (self.tile_aspect, self.columns):
            widget.blockSignals(False)
        self._refresh_layout_info()

    def _apply_layout(self) -> None:
        layout = self.session.project.layout.replace(
            tile_aspect=self.tile_aspect.currentData(),
            columns=self.columns.value(),
        )
        self.session.set_layout(layout)

    def _refresh_layout_info(self) -> None:
        layout = self.session.project.layout
        final = self.session.project.source_final
        if final is None:
            for label in (self._rows, self._source_px):
                label.setText("—")
            return
        src_h, src_w = final.shape[:2]
        rows = layout.rows(src_w, src_h)
        whole = layout.whole_rows(src_w, src_h)
        exact = math.isclose(rows, whole, abs_tol=1e-6)
        self._rows.setText(f"{whole:,}" if exact else f"{rows:,.2f} ({whole:,} whole)")
        per_tile = layout.source_per_tile(src_w)
        self._source_px.setText(f"{per_tile:,.1f} px")
        self._source_px.setStyleSheet(WARNING_STYLE if per_tile < MIN_SOURCE_PX_PER_TILE else "")

    # Plan editing

    def current_row(self) -> int:
        return self._stages.currentRow()

    def add_stage(self, operation: SlicingOperation) -> None:
        """Insert a stage after the selected one (or at the end)."""
        row = self.current_row()
        index = row + 1 if row >= 0 else len(self.plan.stages)
        self.plan.stages.insert(index, Stage(operation))
        self._refresh_stages(select=index)
        self._schedule()

    def remove_stage(self) -> None:
        row = self.current_row()
        if row < 0:
            return
        del self.plan.stages[row]
        self._refresh_stages(select=min(row, len(self.plan.stages) - 1))
        self._schedule()

    def move_stage(self, delta: int) -> None:
        row, stages = self.current_row(), self.plan.stages
        target = row + delta
        if row < 0 or not 0 <= target < len(stages):
            return
        stages[row], stages[target] = stages[target], stages[row]
        self._refresh_stages(select=target)
        self._schedule()

    def _refresh_stages(self, select: int) -> None:
        self._stages.blockSignals(True)
        self._stages.clear()
        for stage in self.plan.stages:
            item = QListWidgetItem(_stage_text(stage))
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Checked if stage.enabled else Qt.CheckState.Unchecked)
            item.setToolTip(stage.operation.description)
            self._stages.addItem(item)
        self._stages.setCurrentRow(select)
        self._stages.blockSignals(False)
        self._on_stage_selected(self.current_row())

    def _on_stage_selected(self, row: int) -> None:
        stages = self.plan.stages
        operation = stages[row].operation if 0 <= row < len(stages) else None
        self._settings_group.setTitle(f"{operation.name} Settings" if operation else "Settings")
        self._description.setText(operation.description if operation else "No stage selected.")
        self.form.set_target(operation)
        self._remove_button.setEnabled(operation is not None)
        self._up_button.setEnabled(row > 0)
        self._down_button.setEnabled(0 <= row < len(stages) - 1)

    def _on_stage_toggled(self, item: QListWidgetItem) -> None:
        stage = self.plan.stages[self._stages.row(item)]
        stage.enabled = item.checkState() == Qt.CheckState.Checked
        self._schedule()

    def _on_param_changed(self, name: str) -> None:
        row = self.current_row()
        item = self._stages.item(row)
        self._stages.blockSignals(True)
        item.setText(_stage_text(self.plan.stages[row]))
        self._stages.blockSignals(False)
        self._schedule()

    def _schedule(self) -> None:
        self._recompute.start()

    # Session updates

    def _on_source_committed(self) -> None:
        self.viewer.show_image(self.session.project.source_final, self.session.mosaic_size())
        self._refresh_layout_info()
        self._sync_preview()

    def _on_layout_changed(self) -> None:
        self._refresh_layout_info()
        size = self.session.mosaic_size()
        if size is not None and self.viewer.world_size != size:
            self.viewer.set_world_size(*size)
        self._sync_preview()

    def _sync_preview(self) -> None:
        """Point the overlay at the image and fit the view around overhanging regions."""
        size = self.session.mosaic_size()
        if size is None or self.viewer.image_layer is None:
            return
        self.overlay.set_image(self.viewer.image_layer, *size)
        regions = self.session.project.regions
        if regions:
            bounds = regions.bounds()
            x0, y0 = bounds[:, :2].min(axis=0)
            x1, y1 = bounds[:, 2:].max(axis=0)
            self.viewer.set_content_bounds((x0, y0, x1 - x0, y1 - y0))

    def _on_slicing_changed(self) -> None:
        regions = self.session.project.regions
        summary = self.session.slicing_summary
        self.overlay.set_regions(regions)
        self._sync_preview()
        error = self.session.slicing_error
        self._error.setText(error or "")
        self._error.setVisible(bool(error))
        if regions and summary:
            self._count.setText(f"{summary.count:,}")
            self._density.setText(f"{summary.density:,.2f}×")
            self._coverage.setText(f"{summary.coverage * 100:.1f}%")
            self._coverage.setStyleSheet(WARNING_STYLE if summary.coverage < 0.999 else "")
            for label, (w, h) in (
                (self._smallest, summary.smallest),
                (self._median, summary.median),
                (self._largest, summary.largest),
            ):
                label.setText(self._in_tiles(w, h))
        else:
            for label in (
                self._count, self._density, self._coverage,
                self._smallest, self._median, self._largest,
            ):  # fmt: skip
                label.setText("—")
            self._coverage.setStyleSheet("")
        self._hover.setText("—")
        self.state_changed.emit()

    def _on_hover(self, x: float | None, y: float | None) -> None:
        regions = self.session.project.regions
        index = -1 if x is None or not regions else regions.hit_test(x, y)
        if index < 0:
            self.overlay.set_highlight(None)
            self._hover.setText("—")
            return
        self.overlay.set_highlight(index)
        region = regions[index]
        layer = int(regions.stacking_rank()[index]) + 1
        self._hover.setText(
            f"#{index + 1:,}: {self._in_tiles(region.width, region.height)}, "
            f"{math.degrees(region.rotation):+.1f}°, layer {layer:,} of {len(regions):,}"
        )

    def _in_tiles(self, width: float, height: float) -> str:
        """A region size in base tiles (1 × 1 is one base tile)."""
        ctx = self.session.slice_context
        if ctx is None:
            return "—"
        tile_w, tile_h = ctx.tile_size
        return f"{width / tile_w:,.2f} × {height / tile_h:,.2f} tiles"

    def is_complete(self) -> bool:
        return bool(self.session.project.regions)
