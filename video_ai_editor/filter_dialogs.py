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
    """Dialog for filtering by confidence level"""
    def __init__(self, scene, parent=None):
        super().__init__(parent)
        self.scene = scene
        self.setWindowTitle("Confidence Filter")
        self.setModal(False)
        self.resize(400, 300)
        self.init_ui()

    def init_ui(self):
        """Initialize the confidence filter UI"""
        layout = QVBoxLayout(self)
        
        # Title
        title = QLabel("Filter by Confidence Level")
        title.setStyleSheet("font-weight: bold; font-size: 14px; color: #a0c0ff;")
        layout.addWidget(title)
        
        # Description
        desc = QLabel("Adjust the confidence range to filter actions and objects.\nConfidence range: 0.0 (low) to 1.0 (high)")
        desc.setStyleSheet("color: #cccccc; font-size: 11px;")
        desc.setWordWrap(True)
        layout.addWidget(desc)
        
        # Current range display
        self.range_label = QLabel(f"Current: {self.scene.min_confidence:.2f} - {self.scene.max_confidence:.2f}")
        self.range_label.setStyleSheet("font-weight: bold; color: #a0ffa0;")
        layout.addWidget(self.range_label)
        
        # Minimum confidence slider
        min_group = QGroupBox("Minimum Confidence")
        min_layout = QVBoxLayout()
        
        self.min_slider = QSlider(Qt.Horizontal)
        self.min_slider.setRange(0, 100)
        self.min_slider.setValue(int(self.scene.min_confidence * 100))
        self.min_slider.valueChanged.connect(self.on_slider_changed)
        
        self.min_value_label = QLabel(f"{self.scene.min_confidence:.2f}")
        self.min_value_label.setStyleSheet("color: #ffa0a0;")
        
        min_layout.addWidget(self.min_slider)
        min_layout.addWidget(self.min_value_label)
        min_group.setLayout(min_layout)
        layout.addWidget(min_group)
        
        # Maximum confidence slider
        max_group = QGroupBox("Maximum Confidence")
        max_layout = QVBoxLayout()
        
        self.max_slider = QSlider(Qt.Horizontal)
        self.max_slider.setRange(0, 100)
        self.max_slider.setValue(int(self.scene.max_confidence * 100))
        self.max_slider.valueChanged.connect(self.on_slider_changed)
        
        self.max_value_label = QLabel(f"{self.scene.max_confidence:.2f}")
        self.max_value_label.setStyleSheet("color: #ffa0a0;")
        
        max_layout.addWidget(self.max_slider)
        max_layout.addWidget(self.max_value_label)
        max_group.setLayout(max_layout)
        layout.addWidget(max_group)
        
        # Preset buttons
        preset_layout = QHBoxLayout()
        
        high_btn = QPushButton("High (0.7-1.0)")
        high_btn.clicked.connect(lambda: self.set_range(0.7, 1.0))
        
        medium_btn = QPushButton("Medium (0.4-0.7)")
        medium_btn.clicked.connect(lambda: self.set_range(0.4, 0.7))
        
        low_btn = QPushButton("Low (0.0-0.4)")
        low_btn.clicked.connect(lambda: self.set_range(0.0, 0.4))
        
        all_btn = QPushButton("All (0.0-1.0)")
        all_btn.clicked.connect(lambda: self.set_range(0.0, 1.0))
        
        preset_layout.addWidget(high_btn)
        preset_layout.addWidget(medium_btn)
        preset_layout.addWidget(low_btn)
        preset_layout.addWidget(all_btn)
        layout.addLayout(preset_layout)
        
        # Statistics
        self.stats_label = QLabel()
        self.stats_label.setStyleSheet("color: #cccccc; font-size: 10px; margin-top: 10px;")
        self.stats_label.setWordWrap(True)
        layout.addWidget(self.stats_label)
        
        # Update initial statistics
        self.update_statistics()
        
        # Dialog buttons
        buttons = QDialogButtonBox(
            QDialogButtonBox.Apply | QDialogButtonBox.Close
        )
        buttons.button(QDialogButtonBox.Apply).clicked.connect(self.apply_filters)
        buttons.button(QDialogButtonBox.Close).clicked.connect(self.close)
        layout.addWidget(buttons)
        
        # Apply dark theme
        self.setStyleSheet("""
            QDialog {
                background-color: #1a1a2a;
            }
            QGroupBox {
                color: #d0e0ff;
                border: 1px solid #3a3a50;
                border-radius: 6px;
                margin-top: 14px;
                padding-top: 10px;
                font-weight: bold;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 6px;
            }
            QSlider::groove:horizontal {
                height: 8px;
                background: #3a3a5a;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #3a5fcd;
                width: 18px;
                margin: -5px 0;
                border-radius: 9px;
            }
            QPushButton {
                background-color: #2a2a44;
                color: white;
                border: 1px solid #4a4a6a;
                padding: 6px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #3a3a5c;
            }
        """)



    def on_slider_changed(self):
        """Update labels when sliders change"""
        min_val = self.min_slider.value() / 100.0
        max_val = self.max_slider.value() / 100.0
        
        # Ensure min <= max
        if min_val > max_val:
            self.max_slider.setValue(int(min_val * 100))
            max_val = min_val
        
        self.min_value_label.setText(f"{min_val:.2f}")
        self.max_value_label.setText(f"{max_val:.2f}")
        self.range_label.setText(f"Current: {min_val:.2f} - {max_val:.2f}")
        
        # Update statistics preview (without applying)
        self.update_statistics()

    def set_range(self, min_val, max_val):
        """Set range from preset"""
        self.min_slider.setValue(int(min_val * 100))
        self.max_slider.setValue(int(max_val * 100))
        self.on_slider_changed()

    def update_statistics(self):
        """Update filter statistics preview"""
        min_val = self.min_slider.value() / 100.0
        max_val = self.max_slider.value() / 100.0
        
        # Count items that would be visible with these settings
        total_actions = len(self.scene.cache_data.get('actions', []))
        total_objects = len(self.scene.cache_data.get('objects', []))
        
        visible_actions = 0
        for action in self.scene.cache_data.get('actions', []):
            confidence = action.get('confidence')
            if confidence is not None:
                if confidence > 1.0:
                    confidence = confidence / 10.0
                if min_val <= confidence <= max_val:
                    visible_actions += 1
            else:
                visible_actions += 1  # Count items without confidence
        
        visible_objects = 0
        for obj_data in self.scene.cache_data.get('objects', []):
            confidence = obj_data.get('confidence')
            if confidence is not None:
                if confidence > 1.0:
                    confidence = confidence / 10.0
                if min_val <= confidence <= max_val:
                    visible_objects += 1
            else:
                visible_objects += 1
        
        filtered_out = (total_actions + total_objects) - (visible_actions + visible_objects)
        
        stats_text = f"""
Statistics (Preview):
- Total actions: {total_actions} → Visible: {visible_actions}
- Total objects: {total_objects} → Visible: {visible_objects}
- Filtered out: {filtered_out} items
"""
        self.stats_label.setText(stats_text.strip())

    def apply_filters(self):
        """Apply the confidence filters"""
        min_val = self.min_slider.value() / 100.0
        max_val = self.max_slider.value() / 100.0
        
        self.scene.set_confidence_filter(min_val, max_val)
        
        # Update statistics with actual results
        self.update_statistics()