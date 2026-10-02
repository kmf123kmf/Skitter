"""Step 2: divide the final source image into target regions for tile matching.

The user builds a slicing plan: an ordered list of operations (see
skitter.core.slicing). The page lists the stages, generates a settings form
for the selected one from its declared parameters, and draws the resulting
regions over the image as they change. Regions may overlap: the preview
draws them in stacking order, so upper regions hide what they cover.
"""

import math

from PySide6.QtCore import Qt, QTimer
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
    QStyle,
    QToolButton,
    QVBoxLayout,
)

from skitter.core.slicing import SlicingOperation, SlicingPlan, Stage, operation_types
from skitter.ui.steps.base import StepPage, side_panel
from skitter.ui.widgets.image_viewer import ImageViewer
from skitter.ui.widgets.param_form import ParamForm
from skitter.ui.widgets.region_overlay import OUTLINES, STACKED, RegionOverlay

UNCOVERED_DIMMING = 0.6
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

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self.viewer, stretch=1)
        layout.addWidget(
            side_panel(
                self._build_plan_group(),
                self._build_settings_group(),
                self._build_result_group(),
                self._build_display_group(),
            )
        )

        session.source_committed.connect(self._on_source_committed)
        session.slicing_changed.connect(self._on_slicing_changed)
        self.viewer.canvas.cursor_moved.connect(self._on_hover)
        self.viewer.canvas.cursor_left.connect(lambda: self._on_hover(None, None))
        self._refresh_stages(select=0)
        self._apply_display()

    @property
    def plan(self) -> SlicingPlan:
        return self.session.project.slicing_plan

    # Construction

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

        hint = _muted(QLabel("Stages run top to bottom, starting from the whole image."))
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
        self._size = QLabel("—")
        self._count = QLabel("—")
        self._smallest = QLabel("—")
        self._largest = QLabel("—")
        self._hover = QLabel("—")
        self._hover.setWordWrap(True)
        self._error = QLabel()
        self._error.setWordWrap(True)
        self._error.setStyleSheet("color: #c42b1c;")
        self._error.hide()

        group = QGroupBox("Result")
        form = QFormLayout(group)
        form.addRow("Image:", self._size)
        form.addRow("Regions:", self._count)
        form.addRow("Smallest:", self._smallest)
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

        group = QGroupBox("Display")
        form = QFormLayout(group)
        form.addRow("Regions:", self.display_mode)
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
        # Dimming only reads correctly when regions repaint the image they cover.
        dim = stacked and self.dim_uncovered.isChecked()
        self.viewer.set_dimming(UNCOVERED_DIMMING if dim else 0.0)

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
        image = self.session.project.source_final
        self.viewer.show_image(image)
        h, w = image.shape[:2]
        self.overlay.set_image(self.viewer.image_layer, w, h)
        self._size.setText(f"{w:,} × {h:,} px")

    def _on_slicing_changed(self) -> None:
        regions = self.session.project.regions
        self.overlay.set_regions(regions)
        error = self.session.slicing_error
        self._error.setText(error or "")
        self._error.setVisible(bool(error))
        if regions:
            area = regions.area()
            small, large = regions.size[area.argmin()], regions.size[area.argmax()]
            self._count.setText(f"{len(regions):,}")
            self._smallest.setText(f"{small[0]:,.1f} × {small[1]:,.1f} px")
            self._largest.setText(f"{large[0]:,.1f} × {large[1]:,.1f} px")
        else:
            for label in (self._count, self._smallest, self._largest):
                label.setText("—")
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
            f"#{index + 1:,}: {region.width:,.1f} × {region.height:,.1f} px, "
            f"{math.degrees(region.rotation):+.1f}°, layer {layer:,} of {len(regions):,}"
        )

    def is_complete(self) -> bool:
        return bool(self.session.project.regions)
