from PyQt4.QtGui import QDialog, QDockWidget
from PyQt4.QtCore import Qt
from filters_dock import Ui_filterMenu
from historicalfilter_dock import Ui_historicalMenu
from help_dock import Ui_helpDock
from confusion_dock import Ui_confusionDock
from settings_dock import Ui_settingsDock

class filters_dock(QDialog, Ui_filterMenu):
	def __init__(self, parent=None):
         super(filters_dock, self).__init__(parent)
         QDockWidget.__init__(self)
         self.setWindowFlags(Qt.Window)
         self.setupUi(self)

class historical_dock(QDialog, Ui_historicalMenu):
	def __init__(self, parent=None):
         super(historical_dock, self).__init__(parent)
         QDockWidget.__init__(self)
         self.setWindowFlags(Qt.Window)
         self.setupUi(self)
 

class help_dock(QDialog, Ui_helpDock):
	def __init__(self, parent=None):
         super(help_dock, self).__init__(parent)
         QDockWidget.__init__(self)
         self.setWindowFlags(Qt.Window)
         self.setupUi(self)

class confusion_dock(QDialog, Ui_confusionDock):
	def __init__(self, parent=None):
         super(confusion_dock, self).__init__(parent)
         QDockWidget.__init__(self)
         self.setWindowFlags(Qt.Window)
         self.setupUi(self)
		 
class settings_dock(QDialog, Ui_settingsDock):
	def __init__(self, parent=None):
         super(settings_dock, self).__init__(parent)
         QDockWidget.__init__(self)
         self.setWindowFlags(Qt.Window)
         self.setupUi(self)

#confusion_dock.kappa.setText('Hello!')