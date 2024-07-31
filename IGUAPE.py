import sys, time, threading, os
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QFileDialog, QDialog, QProgressDialog
from PyQt5.QtCore import Qt, QSize, QThread, pyqtSignal
from PyQt5.QtGui import QIcon, QPixmap
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import SpanSelector
from matplotlib.cm import ScalarMappable
from matplotlib.artist import Artist
import pandas as pd
import lmfit as lm
from lmfit.models import VoigtModel, PseudoVoigtModel
from PyQt5.QtWidgets import (
    QApplication, QDialog, QMainWindow, QMessageBox
)
from PyQt5.uic import loadUi
from Monitor import FolderMonitor
from iguape_GUI import Ui_MainWindow
from Monitor import data_read, peak_fit, counter
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtWidgets import QLabel, QProgressDialog, QMessageBox, QVBoxLayout, QWidget
import numpy as np

# --- Defining the storaging lists --- #        

    
counter.count = 0



class Window(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUi(self)
        self.create_graphs_layout()

    def create_graphs_layout(self):
        self.XRD_data_layout = QVBoxLayout()
        # Creating the main plot #
        self.fig_main = Figure(figsize=(8, 6), dpi=100)
        self.gs_main = self.fig_main.add_gridspec(1, 1)
        self.ax_main = self.fig_main.add_subplot(self.gs_main[0, 0])
        self.fig_main.set_layout_engine('compressed')
        self.canvas_main = FigureCanvas(self.fig_main)
        self.XRD_data_layout.addWidget(self.canvas_main)
        self.XRD_data_tab.setLayout(self.XRD_data_layout)
    
        self.peak_fit_layout = QVBoxLayout()
        # Creating the subplots #
        self.fig_sub = Figure(figsize=(8, 5), dpi=100)
        self.gs_sub = self.fig_sub.add_gridspec(1, 3)
        self.ax_2theta = self.fig_sub.add_subplot(self.gs_sub[0, 0])
        self.ax_area = self.fig_sub.add_subplot(self.gs_sub[0, 2])
        self.ax_FWHM = self.fig_sub.add_subplot(self.gs_sub[0, 1])
        self.fig_sub.set_layout_engine('compressed')
        self.canvas_sub = FigureCanvas(self.fig_sub)
        self.peak_fit_layout.addWidget(self.canvas_sub)
        self.peak_fit_tab.setLayout(self.peak_fit_layout)

        # Creating a colormap on the main canvas #
        self.cmap = plt.get_cmap('coolwarm') 
        self.norm = plt.Normalize(vmin=0, vmax=1) # Initial placeholder values for norm #
        self.sm = ScalarMappable(cmap=self.cmap, norm=self.norm)
        self.sm.set_array([])
        self.cax = self.fig_main.colorbar(self.sm, ax=self.ax_main) # Creating the colorbar axes #
        
        # Create button layout #
        self.refresh_button.clicked.connect(self.update_graphs)
        
        self.reset_button.clicked.connect(self.reset_interval)
        
        self.peak_fit_button.clicked.connect(self.select_fit_interval)
        self.save_XRD_figure_button.clicked.connect(self.save_fig_main)

        self.save_peak_fit_button.clicked.connect(self.save_fig_sub)

        self.save_peak_fit_data_button.clicked.connect(self.save_data_frame)
        
        self.plot_with_temp = False
        self.selected_interval = None
        self.fit_interval = None
        self.offset_signal = False
        self.folder_selected = False
        self.temp_mask_signal = False

        # Create span selector on the main plot
        self.span = SpanSelector(self.ax_main, self.onselect, 'horizontal', useblit=True,
                                props=dict(alpha=0.3, facecolor='red', capstyle='round'))
        self.XRD_data_layout.addWidget(NavigationToolbar(self.canvas_main,self))
        self.peak_fit_layout.addWidget(NavigationToolbar(self.canvas_sub, self))
        
        self.actionOpen_New_Folder.triggered.connect(self.select_folder)
        self.actionAbout.triggered.connect(self.about)
        
        self.offset_slider.setMinimum(1)
        self.offset_slider.setMaximum(100)
        self.offset_slider.setValue(10)

        #self.temperature_checkbox.setCheckable(False)
        self.XRD_measure_order_checkbox.stateChanged.connect(self.measure_order_index)
        self.temperature_checkbox.stateChanged.connect(self.temp_index)
        self.filter_button.clicked.connect(self.apply_temp_mask)

    def update_graphs(self):
        # Clear previous plots #
        try:
            self.ax_main.clear()
            self.plot_data = self.monitor.data_frame
        except AttributeError:
            print('Please initialize the monitor')
            pass
        except Exception as e:
            print(f'Exception: {e}. Please verify the error.')
            pass
        try:
            if self.temp_mask_signal:
                self.plot_data = self.plot_data[self.temp_mask].reset_index(drop=True)
        except pd.errors.IndexingError:
            pass

        try:
            if self.plot_with_temp:
                self.norm.vmin = min(self.plot_data['temp']) # Updating the normalization of the colormap #
                self.norm.vmax = max(self.plot_data['temp'])
                self.sm.set_norm(self.norm)
                self.cax.set_label('Temperature (°C)')
                self.min_temp_doubleSpinBox.setRange(min(self.monitor.data_frame['temp']), max(self.monitor.data_frame['temp']))
                self.min_temp_doubleSpinBox.setValue(min(self.monitor.data_frame['temp']))
                self.max_temp_doubleSpinBox_2.setRange(min(self.monitor.data_frame['temp']), max(self.monitor.data_frame['temp']))
                self.max_temp_doubleSpinBox_2.setValue(max(self.monitor.data_frame['temp']))
            else:
                self.norm.vmin = min(self.plot_data['file_index'])
                self.norm.vmax = max(self.plot_data['file_index'])
                self.sm.set_norm(self.norm)
                self.cax.set_label('XRD measure order')
                self.min_temp_doubleSpinBox.setRange(min(self.monitor.data_frame['file_index']), max(self.monitor.data_frame['file_index']))
                self.min_temp_doubleSpinBox.setValue(min(self.monitor.data_frame['file_index']))
                self.max_temp_doubleSpinBox_2.setRange(min(self.monitor.data_frame['file_index']), max(self.monitor.data_frame['file_index']))
                self.max_temp_doubleSpinBox_2.setValue(max(self.monitor.data_frame['file_index']))
            
            self.spacing = max(self.plot_data['max'])/self.offset_slider.value() # Defining curves offset #
            offset = 0
            for i in range(len(self.plot_data['theta'])):
                if self.selected_interval and self.plot_with_temp:
                    self.mask = (self.plot_data['theta'][i] >= self.selected_interval[0]) & (self.plot_data['theta'][i] <= self.selected_interval[1]) # mask defined by the span tool #
                    self.ax_main.plot(self.plot_data['theta'][i][self.mask], self.plot_data['intensity'][i][self.mask] + offset, color = self.cmap(self.norm(self.plot_data['temp'][i])))
            
                elif self.selected_interval:
                    self.mask = (self.plot_data['theta'][i] >= self.selected_interval[0]) & (self.plot_data['theta'][i] <= self.selected_interval[1]) # mask defined by the span tool #
                    self.ax_main.plot(self.plot_data['theta'][i][self.mask], self.plot_data['intensity'][i][self.mask] + offset, color = self.cmap(self.norm(self.plot_data['file_index'][i])))

                elif self.plot_with_temp:
                    self.ax_main.plot(self.plot_data['theta'][i], self.plot_data['intensity'][i] + offset, color = self.cmap(self.norm(self.plot_data['temp'][i])))
            
                else:
                    self.ax_main.plot(self.plot_data['theta'][i], self.plot_data['intensity'][i] + offset, color = self.cmap(self.norm(self.plot_data['file_index'][i])))

                offset+=self.spacing
            
            self.ax_main.set_xlabel('2θ (°)')
            self.ax_main.set_ylabel('Intensity (u.a.)')

        # Plotting fitting parameters on the sub figure #
            if self.fit_interval and self.plot_with_temp:
                self.ax_2theta.clear()
                self.ax_2theta.plot(self.monitor.fit_data['temp'], self.monitor.fit_data['dois_theta_0'], 'ro')
                self.ax_2theta.set_xlabel('Blower Temperature (°C)')
                self.ax_2theta.set_ylabel('Peak position (°)')
            
                self.ax_area.clear()
                self.ax_area.plot(self.monitor.fit_data['temp'], self.monitor.fit_data['area'], 'go')
                self.ax_area.set_xlabel('Blower Temperature (°C)')
                self.ax_area.set_ylabel('Peak integrated area')
            
                self.ax_FWHM.clear()
                self.ax_FWHM.plot(self.monitor.fit_data['temp'], self.monitor.fit_data['fwhm'], 'bo')
                self.ax_FWHM.set_xlabel('Blower Temperature (°C)')
                self.ax_FWHM.set_ylabel('FWHM (°)')

            #Add shade to fitting interval
                self.ax_main.axvspan(self.fit_interval[0], self.fit_interval[1], color='grey', alpha=0.5, label='Selected Fitting Interval')
                self.ax_main.legend()

            elif self.fit_interval and not self.plot_with_temp:
                self.ax_2theta.clear()
                self.ax_2theta.plot(self.monitor.fit_data['file_index'], self.monitor.fit_data['dois_theta_0'], 'ro')
                self.ax_2theta.set_xlabel('XRD index')
                self.ax_2theta.set_ylabel('Peak position (°)')
            
                self.ax_area.clear()
                self.ax_area.plot(self.monitor.fit_data['file_index'], self.monitor.fit_data['area'], 'go')
                self.ax_area.set_xlabel('XRD index')
                self.ax_area.set_ylabel('Peak integrated area')
            
                self.ax_FWHM.clear()
                self.ax_FWHM.plot(self.monitor.fit_data['file_index'], self.monitor.fit_data['fwhm'], 'bo')
                self.ax_FWHM.set_xlabel('XRD index')
                self.ax_FWHM.set_ylabel('FWHM (°)')

            #Add shade to fitting interval
                self.ax_main.axvspan(self.fit_interval[0], self.fit_interval[1], color='grey', alpha=0.5, label='Selected Fitting Interval') # Shaded area on main figure to highlight fitting interval
                self.ax_main.legend()

        # Update canvas #
            self.canvas_main.draw()
            self.canvas_sub.draw()
            self.cax.update_normal(self.sm)
            
        except Exception as e:
            print(f'No files encountered yet. Execption {e}')
            pass

    def select_folder(self):
        if self.folder_selected:
            self.monitor.data_frame = self.monitor.data_frame.iloc[0:0]
            self.monitor.fit_data = self.monitor.fit_data.iloc[0:0]
        counter.count = 0
        self.plot_with_temp = False
        self.selected_interval = None
        self.fit_interval = None
        self.offset_signal = False
        self.folder_selected = False
        self.temp_mask_signal = False
        folder_path = QFileDialog.getExistingDirectory(None, 'Select the data folder to monitor', '/home') # Selection of monitoring folder
        if folder_path:
            print('Selected Folder:', folder_path)
            self.folder_selected = True
            self.monitor = FolderMonitor(folder_path=folder_path)
            monitor_thread = threading.Thread(target=self.monitor.run)
            monitor_thread.daemon=True
            monitor_thread.start()
            
        else:
            print('No folder selected. Exiting')
            


    def onselect(self, xmin, xmax):
        self.selected_interval = (xmin, xmax)
        #print(f'Selected Interval: {self.selected_interval}')
        self.update_graphs()
    # Reset button function #     
    def reset_interval(self):
        self.selected_interval = None
        self.update_graphs()
    # Peak fit interval selection routine # 
    def select_fit_interval(self):
        try:
            if len(self.plot_data['theta']) == 0:
                print('No data available. Wait for the first measure!')
            else:
                self.ax_2theta.clear()
                self.ax_area.clear()
                self.ax_FWHM.clear()
                self.fit_interval=None
                self.monitor.fit_data = self.monitor.fit_data.iloc[0:0]
                self.fit_interval_window = FitIntervalWindow(self.plot_data['theta'][0], self.plot_data['intensity'][0])  # Calling a second window and passing the data to the new window
                self.fit_interval_window.show()
        except AttributeError:
            print('Please, initialize the monitor!')
    # Peak fit with the new selected interval #        

    def save_fig_main(self):
        try:
            options = QFileDialog.Options()
            file_path, _ = QFileDialog.getSaveFileName(self, "Save Main Figure", "", "PNG Files (*.png);;All Files (*)", options=options)
            if file_path:
                self.fig_main.savefig(fname=file_path)
        except Exception:
            print('No figure available!')
    
    def save_fig_sub(self):
        try:
            options = QFileDialog.Options()
            file_path, _ = QFileDialog.getSaveFileName(self, "Save Fittin Figures", "", "PNG Files (*.png);;All Files (*)", options=options)
            if file_path:
                self.fig_sub.savefig(fname=file_path)
        except Exception:
            print('No figure available')

    def save_data_frame(self):
        try:
            options = QFileDialog.Options()
            if self.plot_with_temp:
                df = pd.DataFrame({'Blower Temperature (°C)': self.monitor.fit_data['temp'], 'Peak position (degree)': self.monitor.fit_data['dois_theta_0'], 'Peak Integrated Area': self.monitor.fit_data['area'], 
                      'FWHM (degree)': self.monitor.fit_data['fwhm'], 'R-squared (R²)': self.monitor.fit_data['R-squared']})
                file_path, _ = QFileDialog.getSaveFileName(self, "Save fitting Data", "", "Excel Files (*.xlsx);;All Files (*)", options=options)
            
                if file_path:
                    df.to_csv(file_path)
            else:
                df = pd.DataFrame({'Measure': self.monitor.fit_data['file_index'], 'Peak position (degree)': self.monitor.fit_data['dois_theta_0'], 'Peak Integrated Area': self.monitor.fit_data['area'], 
                      'FWHM (degree)': self.monitor.fit_data['fwhm'], 'R-squared (R²)': self.monitor.fit_data['R-squared']})
                file_path, _ = QFileDialog.getSaveFileName(self, "Save fitting Data", "", "Excel Files (*.xlsx);;All Files (*)", options=options)
            
                if file_path:
                    df.to_csv(file_path)
        except Exception as e:
            print(f"No data availabe! Exception {e}")

    def validate_temp(self):
        if self.plot_with_temp:
            min_temp = self.min_temp_doubleSpinBox.value()
            max_temp = self.max_temp_doubleSpinBox_2.value()
            temps = self.monitor.data_frame['temp']
            if min_temp not in temps:
                self.min_temp_doubleSpinBox.setValue(min(temps, key=lambda x: abs(x-min_temp)))

            if max_temp not in temps:
                self.max_temp_doubleSpinBox_2.setValue(min(temps, key=lambda x: abs(x-max_temp)))
            return min_temp, max_temp


    def apply_temp_mask(self):
        if self.plot_with_temp:
            min_temp, max_temp = self.validate_temp()
            self.temp_mask = (self.monitor.data_frame['temp'] >= min_temp) & (self.monitor.data_frame['temp'] <= max_temp)
        else:
            self.temp_mask = (self.monitor.data_frame['file_index'] >= self.min_temp_doubleSpinBox.value()) & (self.monitor.data_frame['file_index'] <= self.max_temp_doubleSpinBox_2.value())
        self.temp_mask_signal = True
        self.update_graphs()

    def measure_order_index(self, checked):
        if checked:
            self.temperature_checkbox.setCheckState(False)
            self.plot_with_temp = False
            self.min_filter_label.setText('Minimum')
            self.max_filter_label.setText('Maximum')
            self.update_graphs()
        else:
            self.temperature_checkbox.setCheckable(True)

    def temp_index(self, checked):
        try:
            if checked:
                if self.monitor.data_frame['temp'][0] != None:
                    self.XRD_measure_order_checkbox.setCheckState(False)
                    self.plot_with_temp = True
                    self.min_filter_label.setText('Minimum Temperature')
                    self.max_filter_label.setText('Maximum Temperature')
                    self.update_graphs()
                else:
                    print("This experiment doesn't make use of temperature!")
                    self.temperature_checkbox.setCheckState(False)
            else:
                pass
                
        except AttributeError:
            self.temperature_checkbox.setCheckState(False)
            print('Please initizalie the Monitor')       
            


        
    def about(self):
        QMessageBox.about(
            self,
            "About Iguape",
            "<p>This is the Paineira Graphical User Interface</p>"
            "<p>- Its usage is resttricted to data acquired via in-situ experiments</p>"
            "<p>- Paineira Beamline</p>"
            "<p>- LNLS - CNPEM</p>",
        )





class FitIntervalWindow(QWidget):
    def __init__(self, theta, intensity):
        super().__init__()
        self.setWindowTitle("Select 2θ Interval for Peak Fitting")
    
        self.fig = Figure(figsize=(5, 3), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.ax.plot(theta, intensity)
        self.ax.set_xlabel("2θ (°)")
        self.ax.set_ylabel("Intensity (u.a.)")
        self.canvas = FigureCanvas(self.fig)
        self.span = SpanSelector(self.ax, self.onselect, 'horizontal', useblit=True,
                                props=dict(alpha=0.3, facecolor='red', capstyle='round'))
        # Create layout
        self.toolbar = NavigationToolbar(self.canvas, self)
        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        layout.addWidget(self.toolbar)
        self.setLayout(layout)

    def onselect(self, xmin, xmax):
        fit_interval = [xmin, xmax]
        win.monitor.set_fit_interval(fit_interval)
        win.fit_interval = fit_interval
        win.fit_interval_window.close()

        self.progress_dialog = QProgressDialog("Fitting peaks...", "Cancel", 0, 100, self)
        self.progress_dialog.setWindowModality(Qt.WindowModal)
        self.progress_dialog.setAutoClose(True)
        self.progress_dialog.show()

        # Start the worker thread for peak fitting
        self.worker = Worker(fit_interval)
        self.worker.progress.connect(self.update_progress)
        self.worker.finished.connect(self.peak_fitting_finished)
        self.worker.error.connect(self.peak_fitting_error)
        self.worker.start()

    def update_progress(self, value):
        self.progress_dialog.setValue(value)

    def peak_fitting_finished(self, dois_theta, area, fwhm):
        self.progress_dialog.setValue(100)
        QMessageBox.information(self, "Peak Fitting", "Peak fitting completed successfully!")

        win.update_graphs()
        self.close()

    def peak_fitting_error(self, error_message):
        self.progress_dialog.cancel()
        QMessageBox.warning(self, "Peak Fitting Error", error_message)
        self.show()

class Worker(QThread):
    progress = pyqtSignal(int)
    finished = pyqtSignal(object, object, object) 
    error = pyqtSignal(str) # Changed to emit multiple arrays

    def __init__(self, interval):
        super().__init__()
        self.fit_interval = interval

    def run(self):
        try:
            for i in range(len(win.plot_data['theta'])):
                fit = peak_fit(win.plot_data['theta'][i], win.plot_data['intensity'][i], self.fit_interval)
                new_fit_data = pd.DataFrame({'dois_theta_0': [fit[0]], 'fwhm': [fit[1]], 'area': [fit[2]], 'temp': [win.plot_data['temp'][i]], 'file_index': [win.plot_data['file_index'][i]], 'R-squared': [fit[3]]})
                win.monitor.fit_data = pd.concat([win.monitor.fit_data, new_fit_data], ignore_index=True)
                progress_value = int((i + 1) / len(win.plot_data['theta']) * 100)
                self.progress.emit(progress_value)  # Emit progress signal with percentage
            self.finished.emit(win.monitor.fit_data['dois_theta_0'], win.monitor.fit_data['area'], win.monitor.fit_data['fwhm'])  # Emit finished signal with results
        except Exception as e:
            self.error.emit(f'Error during peak fitting: {str(e)}. Please select a new Fit Interval')
            #print(f'Exception {e}. Please select a new Fit Interval')



if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = Window()
    win.show()
    sys.exit(app.exec())