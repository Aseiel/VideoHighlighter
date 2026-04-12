from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QListWidget,
    QListWidgetItem, QLineEdit, QCheckBox, QGroupBox,
    QSlider, QPushButton, QDialogButtonBox, QTabWidget, QWidget
)
from PySide6.QtCore import Qt

class FilterDialog(QDialog):
    """Dialog for filtering actions and objects"""
    
    def __init__(self, scene, parent=None):
        super().__init__(parent)
        self.scene = scene
        self.setWindowTitle("Filter Actions & Objects")
        self.setModal(False)  # Non-modal so users can keep it open
        self.resize(500, 600)
        
        self.init_ui()
    
    def init_ui(self):
        layout = QVBoxLayout(self)
        
        # Create tab widget
        tabs = QTabWidget()
        
        # Actions tab
        actions_tab = QWidget()
        actions_layout = QVBoxLayout(actions_tab)
        
        # Actions filter controls
        actions_header = QLabel("Filter Actions")
        actions_header.setStyleSheet("font-weight: bold; font-size: 14px; color: #a0c0ff;")
        actions_layout.addWidget(actions_header)
        
        # Search box for actions
        self.action_search = QLineEdit()
        self.action_search.setPlaceholderText("Search actions...")
        self.action_search.textChanged.connect(self.filter_action_list)
        actions_layout.addWidget(self.action_search)
        
        # Actions list
        self.action_list = QListWidget()
        self.action_list.setSelectionMode(QListWidget.MultiSelection)
        self.populate_action_list()
        actions_layout.addWidget(self.action_list)
        
        # Action buttons
        action_buttons = QHBoxLayout()
        self.select_all_actions = QPushButton("Select All")
        self.select_all_actions.clicked.connect(lambda: self.set_all_actions(True))
        self.deselect_all_actions = QPushButton("Deselect All")
        self.deselect_all_actions.clicked.connect(lambda: self.set_all_actions(False))
        
        action_buttons.addWidget(self.select_all_actions)
        action_buttons.addWidget(self.deselect_all_actions)
        actions_layout.addLayout(action_buttons)
        
        tabs.addTab(actions_tab, "Actions")
        
        # Objects tab
        objects_tab = QWidget()
        objects_layout = QVBoxLayout(objects_tab)
        
        # Objects filter controls
        objects_header = QLabel("Filter Objects")
        objects_header.setStyleSheet("font-weight: bold; font-size: 14px; color: #ffa0a0;")
        objects_layout.addWidget(objects_header)
        
        # Search box for objects
        self.object_search = QLineEdit()
        self.object_search.setPlaceholderText("Search objects...")
        self.object_search.textChanged.connect(self.filter_object_list)
        objects_layout.addWidget(self.object_search)
        
        # Objects list
        self.object_list = QListWidget()
        self.object_list.setSelectionMode(QListWidget.MultiSelection)
        self.populate_object_list()
        objects_layout.addWidget(self.object_list)
        
        # Object buttons
        object_buttons = QHBoxLayout()
        self.select_all_objects = QPushButton("Select All")
        self.select_all_objects.clicked.connect(lambda: self.set_all_objects(True))
        self.deselect_all_objects = QPushButton("Deselect All")
        self.deselect_all_objects.clicked.connect(lambda: self.set_all_objects(False))
        
        object_buttons.addWidget(self.select_all_objects)
        object_buttons.addWidget(self.deselect_all_objects)
        objects_layout.addLayout(object_buttons)
        
        tabs.addTab(objects_tab, "Objects")
        
        layout.addWidget(tabs)
        
        # Dialog buttons
        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Apply | QDialogButtonBox.Cancel
        )
        buttons.accepted.connect(self.apply_and_close)
        buttons.button(QDialogButtonBox.Apply).clicked.connect(self.apply_filters)
        buttons.rejected.connect(self.reject)
        
        layout.addWidget(buttons)
        
        # Apply dark theme
        self.setStyleSheet("""
            QDialog {
                background-color: #1a1a2a;
            }
            QListWidget {
                background-color: #2a2a3a;
                color: #ffffff;
                border: 1px solid #444466;
                border-radius: 4px;
            }
            QListWidget::item {
                padding: 5px;
                border-bottom: 1px solid #333344;
            }
            QListWidget::item:selected {
                background-color: #3a5fcd;
            }
            QListWidget::item:hover {
                background-color: #3a3a5a;
            }
            QLineEdit {
                background-color: #2a2a3a;
                color: #ffffff;
                border: 1px solid #444466;
                border-radius: 4px;
                padding: 5px;
            }
            QTabWidget::pane {
                border: 1px solid #444466;
                background-color: #2a2a3a;
            }
            QTabBar::tab {
                background-color: #3a3a5a;
                color: #cccccc;
                padding: 8px 16px;
                margin-right: 2px;
            }
            QTabBar::tab:selected {
                background-color: #4a5fcd;
                color: #ffffff;
            }
        """)
    
    def populate_action_list(self):
        """Populate action list with all action types"""
        self.action_list.clear()
        for action in sorted(self.scene.action_types):
            item = QListWidgetItem(action)
            item.setCheckState(Qt.Checked if self.scene.visible_actions.get(action, True) else Qt.Unchecked)
            self.action_list.addItem(item)
    
    def populate_object_list(self):
        """Populate object list with all object types"""
        self.object_list.clear()
        for obj in sorted(self.scene.object_classes):
            item = QListWidgetItem(obj)
            item.setCheckState(Qt.Checked if self.scene.visible_objects.get(obj, True) else Qt.Unchecked)
            self.object_list.addItem(item)
    
    def filter_action_list(self):
        """Filter action list based on search text"""
        search_text = self.action_search.text().lower()
        for i in range(self.action_list.count()):
            item = self.action_list.item(i)
            item.setHidden(search_text not in item.text().lower())
    
    def filter_object_list(self):
        """Filter object list based on search text"""
        search_text = self.object_search.text().lower()
        for i in range(self.object_list.count()):
            item = self.object_list.item(i)
            item.setHidden(search_text not in item.text().lower())
    
    def set_all_actions(self, selected):
        """Select or deselect all actions"""
        for i in range(self.action_list.count()):
            item = self.action_list.item(i)
            if not item.isHidden():
                item.setCheckState(Qt.Checked if selected else Qt.Unchecked)
    
    def set_all_objects(self, selected):
        """Select or deselect all objects"""
        for i in range(self.object_list.count()):
            item = self.object_list.item(i)
            if not item.isHidden():
                item.setCheckState(Qt.Checked if selected else Qt.Unchecked)
    
    def apply_filters(self):
        """Apply the selected filters to the scene"""
        # Update action filters
        for i in range(self.action_list.count()):
            item = self.action_list.item(i)
            action_name = item.text()
            self.scene.set_action_filter(action_name, item.checkState() == Qt.Checked)
        
        # Update object filters
        for i in range(self.object_list.count()):
            item = self.object_list.item(i)
            object_name = item.text()
            self.scene.set_object_filter(object_name, item.checkState() == Qt.Checked)
    
    def apply_and_close(self):
        """Apply filters and close dialog"""
        self.apply_filters()
        self.accept()

class ConfidenceFilterDialog(QDialog):
    """Dialog for filtering by confidence level — separate for actions and objects"""
    def __init__(self, scene, parent=None):
        super().__init__(parent)
        self.scene = scene
        self.setWindowTitle("Confidence Filter")
        self.setModal(False)
        self.resize(450, 400)
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)

        title = QLabel("Filter by Confidence Level")
        title.setStyleSheet("font-weight: bold; font-size: 14px; color: #a0c0ff;")
        layout.addWidget(title)

        desc = QLabel("Adjust minimum confidence separately for actions and objects.")
        desc.setStyleSheet("color: #cccccc; font-size: 11px;")
        desc.setWordWrap(True)
        layout.addWidget(desc)

        # ── Actions confidence ──
        action_group = QGroupBox("🎬 Action Confidence")
        action_layout = QVBoxLayout()

        self.action_slider = QSlider(Qt.Horizontal)
        self.action_slider.setRange(0, 100)
        self.action_slider.setValue(int(self.scene.min_action_confidence * 100))
        self.action_slider.valueChanged.connect(self.on_slider_changed)

        self.action_label = QLabel(f"{self.scene.min_action_confidence:.0%}")
        self.action_label.setStyleSheet("color: #a0ffa0; font-weight: bold;")

        action_layout.addWidget(self.action_slider)
        action_layout.addWidget(self.action_label)
        action_group.setLayout(action_layout)
        layout.addWidget(action_group)

        # ── Objects confidence ──
        object_group = QGroupBox("📦 Object Confidence")
        object_layout = QVBoxLayout()

        self.object_slider = QSlider(Qt.Horizontal)
        self.object_slider.setRange(0, 100)
        self.object_slider.setValue(int(self.scene.min_object_confidence * 100))
        self.object_slider.valueChanged.connect(self.on_slider_changed)

        self.object_label = QLabel(f"{self.scene.min_object_confidence:.0%}")
        self.object_label.setStyleSheet("color: #ffa0a0; font-weight: bold;")

        object_layout.addWidget(self.object_slider)
        object_layout.addWidget(self.object_label)
        object_group.setLayout(object_layout)
        layout.addWidget(object_group)

        # ── Presets ──
        preset_layout = QHBoxLayout()

        high_btn = QPushButton("High (70%)")
        high_btn.clicked.connect(lambda: self.set_both(70))
        medium_btn = QPushButton("Medium (40%)")
        medium_btn.clicked.connect(lambda: self.set_both(40))
        low_btn = QPushButton("Low (10%)")
        low_btn.clicked.connect(lambda: self.set_both(10))
        all_btn = QPushButton("All (0%)")
        all_btn.clicked.connect(lambda: self.set_both(0))

        preset_layout.addWidget(high_btn)
        preset_layout.addWidget(medium_btn)
        preset_layout.addWidget(low_btn)
        preset_layout.addWidget(all_btn)
        layout.addLayout(preset_layout)

        # ── Stats ──
        self.stats_label = QLabel()
        self.stats_label.setStyleSheet("color: #cccccc; font-size: 10px; margin-top: 10px;")
        self.stats_label.setWordWrap(True)
        layout.addWidget(self.stats_label)
        self.update_statistics()

        # ── Buttons ──
        buttons = QDialogButtonBox(QDialogButtonBox.Apply | QDialogButtonBox.Close)
        buttons.button(QDialogButtonBox.Apply).clicked.connect(self.apply_filters)
        buttons.button(QDialogButtonBox.Close).clicked.connect(self.close)
        layout.addWidget(buttons)

        self.setStyleSheet("""
            QDialog { background-color: #1a1a2a; }
            QGroupBox {
                color: #d0e0ff; border: 1px solid #3a3a50;
                border-radius: 6px; margin-top: 14px; padding-top: 10px;
                font-weight: bold;
            }
            QGroupBox::title { subcontrol-origin: margin; left: 12px; padding: 0 6px; }
            QSlider::groove:horizontal { height: 8px; background: #3a3a5a; border-radius: 4px; }
            QSlider::handle:horizontal { background: #3a5fcd; width: 18px; margin: -5px 0; border-radius: 9px; }
            QPushButton {
                background-color: #2a2a44; color: white;
                border: 1px solid #4a4a6a; padding: 6px; border-radius: 4px;
            }
            QPushButton:hover { background-color: #3a3a5c; }
        """)

    def on_slider_changed(self):
        action_val = self.action_slider.value() / 100.0
        object_val = self.object_slider.value() / 100.0
        self.action_label.setText(f"{action_val:.0%}")
        self.object_label.setText(f"{object_val:.0%}")
        self.update_statistics()

    def set_both(self, percent):
        self.action_slider.setValue(percent)
        self.object_slider.setValue(percent)

    def update_statistics(self):
        action_min = self.action_slider.value() / 100.0
        object_min = self.object_slider.value() / 100.0

        total_actions = len(self.scene.cache_data.get('actions', []))
        total_objects = len(self.scene.cache_data.get('objects', []))

        visible_actions = 0
        for action in self.scene.cache_data.get('actions', []):
            conf = action.get('confidence')
            if conf is not None:
                if conf > 1.0:
                    conf = conf / 10.0
                if conf >= action_min:
                    visible_actions += 1
            else:
                visible_actions += 1

        visible_objects = 0
        for obj in self.scene.cache_data.get('objects', []):
            conf = obj.get('confidence')
            if conf is not None:
                if conf > 1.0:
                    conf = conf / 10.0
                if conf >= object_min:
                    visible_objects += 1
            else:
                visible_objects += 1

        self.stats_label.setText(
            f"Actions: {visible_actions}/{total_actions} visible (≥{action_min:.0%})\n"
            f"Objects: {visible_objects}/{total_objects} visible (≥{object_min:.0%})"
        )

    def apply_filters(self):
        action_min = self.action_slider.value() / 100.0
        object_min = self.object_slider.value() / 100.0
        self.scene.set_action_confidence_filter(action_min)
        self.scene.set_object_confidence_filter(object_min)
        self.update_statistics()