# -*- coding: utf-8 -*-

"""
Anniversary popup widget for dzetsaka 10th anniversary
Created for May 17, 2026 anniversary celebration
"""

import datetime
import webbrowser
from PyQt5 import QtWidgets
from PyQt5.QtCore import QSettings, pyqtSignal
from .anniversary import Ui_AnniversaryDialog


class AnniversaryDialog(QtWidgets.QDialog, Ui_AnniversaryDialog):
    """Anniversary celebration dialog for dzetsaka 10th anniversary"""
    
    closingPlugin = pyqtSignal()

    def __init__(self, parent=None):
        super(AnniversaryDialog, self).__init__(parent)
        self.setupUi(self)
        
        # Update dynamic content based on current date
        self.update_dynamic_content()
        
        # Connect signals
        self.githubButton.clicked.connect(self.open_github_features)
        self.pollButton.clicked.connect(self.open_poll)
        
    def update_dynamic_content(self):
        """Update dialog content based on current date"""
        today = datetime.date.today()
        anniversary_date = datetime.date(2026, 5, 17)
        days_until_anniversary = (anniversary_date - today).days
        
        if days_until_anniversary > 0:
            # Still in preparation phase
            time_message = f"We have <b>{days_until_anniversary} days</b> until the anniversary to collect your ideas!"
        elif days_until_anniversary == 0:
            # Anniversary day
            time_message = "🎉 <b>Today is the anniversary!</b> Thank you for all your feature suggestions!"
        else:
            # After anniversary (shouldn't normally happen with our date logic)
            time_message = "Thank you for your continued support of dzetsaka!"
        
        # Update the features label with dynamic content
        current_text = self.featuresLabel.text()
        if "We have" not in current_text and "Today is" not in current_text:
            new_text = current_text + f"<br><br><i>{time_message}</i>"
            self.featuresLabel.setText(new_text)

    def open_github_features(self):
        """Open GitHub issues page for feature requests"""
        webbrowser.open("https://github.com/nkarasiak/dzetsaka/issues")
    
    def open_poll(self):
        """Open poll/survey for structured feedback"""
        from PyQt5.QtWidgets import QMessageBox
        
        # Create a message box with multiple poll options
        msg = QMessageBox(self)
        msg.setWindowTitle("Choose Feedback Method")
        msg.setText("How would you like to share your feedback?")
        msg.setInformativeText(
            "Choose your preferred way to help us plan dzetsaka's future:\n\n"
            "📊 <b>Quick Poll</b> - Structured questions (recommended)\n"
            "💬 <b>GitHub Discussions</b> - Community forum\n"
            "🐛 <b>GitHub Issues</b> - Feature requests & bug reports"
        )
        
        # Custom buttons
        quick_poll_btn = msg.addButton("📊 Quick Poll", QMessageBox.ActionRole)
        discussions_btn = msg.addButton("💬 Discussions", QMessageBox.ActionRole)
        issues_btn = msg.addButton("🐛 Issues", QMessageBox.ActionRole)
        cancel_btn = msg.addButton("Cancel", QMessageBox.RejectRole)
        
        msg.exec_()
        
        if msg.clickedButton() == quick_poll_btn:
            # Open a poll/survey service - you can replace this URL with your actual poll
            # Options: Google Forms, Microsoft Forms, SurveyMonkey, Typeform, etc.
            self._open_quick_poll()
        elif msg.clickedButton() == discussions_btn:
            webbrowser.open("https://github.com/nkarasiak/dzetsaka/discussions")
        elif msg.clickedButton() == issues_btn:
            webbrowser.open("https://github.com/nkarasiak/dzetsaka/issues")
    
    def _open_quick_poll(self):
        """Open the quick poll/survey"""
        # You can replace this with your preferred survey platform
        # Here are some options:
        
        poll_services = {
            "google_forms": "https://forms.gle/YOUR_FORM_ID_HERE",
            "microsoft_forms": "https://forms.office.com/YOUR_FORM_ID",
            "surveymonkey": "https://www.surveymonkey.com/r/YOUR_SURVEY_ID",
            "typeform": "https://YOUR_ACCOUNT.typeform.com/to/YOUR_FORM_ID"
        }
        
        # For now, redirect to GitHub Discussions with poll instructions
        webbrowser.open("https://github.com/nkarasiak/dzetsaka/discussions/new?category=polls")
    
    def closeEvent(self, event):
        self.closingPlugin.emit()
        event.accept()

    def is_dont_show_again_checked(self):
        """Return whether the 'don't show again' checkbox is checked"""
        return self.dontShowAgainCheckBox.isChecked()


class AnniversaryManager:
    """Manages anniversary popup display logic"""
    
    ANNIVERSARY_DATE = datetime.date(2026, 5, 17)  # May 17, 2026
    FEATURE_COLLECTION_START = datetime.date(2025, 7, 17)  # Start collecting features from now
    SETTINGS_KEY_SHOWN = "/dzetsaka/anniversaryShown"
    SETTINGS_KEY_HIDE_FOREVER = "/dzetsaka/anniversaryHideForever"
    
    def __init__(self):
        self.settings = QSettings()
    
    def should_show_anniversary_popup(self):
        """
        Determine if the anniversary popup should be shown
        
        Returns:
            bool: True if popup should be shown, False otherwise
        """
        today = datetime.date.today()
        
        # Check if user has chosen to hide forever
        hide_forever = self.settings.value(self.SETTINGS_KEY_HIDE_FOREVER, False, bool)
        if hide_forever:
            return False
        
        # Check if we're in the feature collection period (from now until anniversary)
        if today < self.FEATURE_COLLECTION_START or today > self.ANNIVERSARY_DATE:
            return False
        
        # Check if already shown today
        last_shown = self.settings.value(self.SETTINGS_KEY_SHOWN, "", str)
        if last_shown == today.isoformat():
            return False
        
        return True
    
    def mark_as_shown(self, hide_forever=False):
        """
        Mark the anniversary popup as shown
        
        Args:
            hide_forever (bool): If True, user chose to hide popup forever
        """
        today = datetime.date.today()
        self.settings.setValue(self.SETTINGS_KEY_SHOWN, today.isoformat())
        
        if hide_forever:
            self.settings.setValue(self.SETTINGS_KEY_HIDE_FOREVER, True)
    
    def show_anniversary_popup(self, parent=None):
        """
        Show the anniversary popup if conditions are met
        
        Args:
            parent: Parent widget for the dialog
            
        Returns:
            bool: True if popup was shown, False otherwise
        """
        if not self.should_show_anniversary_popup():
            return False
        
        dialog = AnniversaryDialog(parent)
        result = dialog.exec_()
        
        # Save user preference
        hide_forever = dialog.is_dont_show_again_checked()
        self.mark_as_shown(hide_forever)
        
        return True
    
    def reset_anniversary_settings(self):
        """Reset anniversary settings (for testing purposes)"""
        self.settings.remove(self.SETTINGS_KEY_SHOWN)
        self.settings.remove(self.SETTINGS_KEY_HIDE_FOREVER)