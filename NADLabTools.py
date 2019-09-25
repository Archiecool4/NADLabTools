"""
NAD Lab Tools

This program was written for the NAD Lab at the University of Arizona by Archie Shahidullah.
It processes intracellular calcium concentration and pH measurements (from the InCytim2 software)
as well as filters the data for outliers and spikes. 

The experiment consists of placing fluorescent-stained cells under a microscope to measure either
calcium concentration or pH. Over a period of time, solutions are added to determine the response
of the quantities.

Archie Shahidullah 2019
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tkinter as tk
from tkinter import ttk
from tkinter.filedialog import askopenfilenames, askdirectory

# File names for analysis
names = []
# Output directory
output = ''
# Names of drugs added to solution
events = []
# Beginning of baseline measurement
itime = 60
# End of baseline measurement
ftime = 200
# Lower threshold to exclude cells
lbase = 50
# Upper threshold to exclude cells
ubase = 150
# Mode to analyze data - can be either 'Calcium' or 'pH'
measure = 'Calcium'


def process_data(df):
    """
    Takes in a pandas dataframe and calculates the mean Calcium/pH as well as ratios
    between different wavelength measurements. Then formats the data into a CSV file.
    Lastly, uses the user-defined thresholds to exclude outlier cells.
    
    Arguments:
        df {pd.DataFrame} -- a dataframe to process
    
    Returns:
        tuple -- a tuple of dataframes containing the processed dataframes, their outliers, and
                 graph data   
    """
    global itime, ftime, lbase, ubase, measure

    # Adjusts parameters based on Calcium/pH mode
    meas = ''
    length1 = ''
    length2 = ''
    meanname = ''
    if measure == 'Calcium':
        meas = 'Ca++'
        length1 = '340'
        length2 = '380'
        meanname = 'Mean Calcium (nM)'
    elif measure == 'pH':
        meas = 'pH'
        length1 = '488'
        length2 = '460'
        meanname = 'Mean pH'

    # Reads pertinent data from dataframe
    times = df.iloc[:, 0].to_frame(name='Time (s)').astype(float)
    calcium = df.filter(like=meas, axis=1).astype(float)
    conc_340 = df.filter(like=length1, axis=1).astype(float)
    conc_380 = df.filter(like=length2, axis=1).astype(float)
    ratio = pd.DataFrame()

    # Calculates ratio of different wavelength measurements
    for i, col in enumerate(conc_340.columns):
        ratio[length1 + '/' + length2 + col[-4:]
              ] = conc_340.iloc[:, i] / conc_380.iloc[:, i]

    # Calculates mean ratio and Calcium/pH
    mean_ratio = ratio.mean(axis=1).to_frame(name='Mean Ratio')
    mean_ca = calcium.mean(axis=1).to_frame(name=meanname)

    # Empty dataframe to space columns in CSV file
    empty = pd.DataFrame(columns=[''])
    # Data for CSV
    processed_data = pd.concat([times, empty, calcium, empty, conc_340,
                                empty, conc_380, empty, ratio, empty, mean_ca, mean_ratio], axis=1)
    # Data for graph
    graph_data = pd.concat([times, mean_ca], axis=1)

    remove = []
    # Get baseline times for cells
    baselines = calcium[(times.iloc[:, 0].astype(float) >= itime) & (
        times.iloc[:, 0].astype(float) <= ftime)]

    # Exclude outliers
    if len(baselines) != 0:
        for i in range(len(baselines.iloc[0])):
            baseline = baselines.iloc[:, i].mean()
            if baseline <= lbase or baseline >= ubase:
                remove.append(i)
    else:
        remove = range(len(calcium.iloc[0]))

    # Compiles outlier data
    calc_outliers = calcium.drop(calcium.columns[remove], axis=1)
    conc_340_outliers = conc_340.drop(conc_340.columns[remove], axis=1)
    conc_380_outliers = conc_380.drop(conc_380.columns[remove], axis=1)
    ratio_outliers = pd.DataFrame()

    # Outlier ratios
    for i, col in enumerate(conc_340_outliers.columns):
        ratio_outliers[length1 + '/' + length2 + col[-4:]] = conc_340_outliers.iloc[:,
                                                                                    i] / conc_380_outliers.iloc[:, i]
    # Outlier means
    mean_ratio_outliers = ratio_outliers.mean(
        axis=1).to_frame(name='Mean Ratio')
    mean_ca_outliers = calc_outliers.mean(axis=1).to_frame(name=meanname)

    # Format CSV for outliers
    processed_outliers = pd.concat([times, empty, calc_outliers, empty, conc_340_outliers, empty,
                                    conc_380_outliers, empty, ratio_outliers, empty, mean_ca_outliers, mean_ratio_outliers], axis=1)
    # Outlier graph
    graph_outliers = pd.concat([times, mean_ca_outliers], axis=1)

    return processed_data, processed_outliers, graph_data, graph_outliers


def save_figure(df, filename, output, events, eventtimes):
    """
    Takes in a pandas dataframe and saves the graph of the data. Also
    labels events on the graph the user defines.
    
    Arguments:
        df {pd.DataFrame} -- a dataframe to generate a graph from
        filename {str} -- the name to save the graph as
        output {str} -- a path to the desired output folder
        events {list} -- a list of events to mark on the diagram
        eventtimes {list} -- a list of the times when the events happened
    """
    global measure

    # Calcium/pH mode
    meanname = ''
    if measure == 'Calcium':
        meanname = 'Mean Calcium (nM)'
    elif measure == 'pH':
        meanname = 'Mean pH'

    # Formatting of the graph
    df.plot(x='Time (s)', y=meanname, kind='line', legend=None)
    ax = plt.gca()
    plt.xlabel('Time (s)')
    plt.ylabel(meanname)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    xdata = ax.lines[0].get_xdata()
    ydata = ax.lines[0].get_ydata()

    # Annotate the plot with events
    for i, event in enumerate(events):
        if i >= len(eventtimes):
            break
        idx = np.where(
            (xdata >= (eventtimes[i] - 3)) & (xdata <= (eventtimes[i] + 3)))
        plt.annotate(event, xy=(xdata[idx], ydata[idx]), xycoords='data', xytext=(0, 50), textcoords='offset points',
                     arrowprops=dict(arrowstyle='-|>', facecolor='black'), horizontalalignment='right', verticalalignment='top')

    plt.savefig(os.path.join(output, filename + '.png'))


def spike_detection(df):
    """
    Implements a simple "bandpass" filter to remove spikes from the data. Also uses
    local median filtering to establish where spikes happen.
    
    Arguments:
        df {pd.DataFrame} -- a dataframe to despike
    
    Returns:
        pd.DataFrame -- the despiked dataframe
    """
    global measure

    # Calcium/pH mode
    meanname = ''
    if measure == 'Calcium':
        meanname = 'Mean Calcium (nM)'
    elif measure == 'pH':
        meanname = 'Mean pH'

    # Filter parameters
    minimum = 15
    maximum = 50
    radius = 5
    epsilon = 10

    # Read means
    means = np.array(df[meanname])
    # Calculate differences in array, e.g. [1, 5, 3] would return [4, -2]
    diffs = np.abs(np.diff(means))
    # Apply "bandpass" filter and flatten array
    remove = np.array(np.where((diffs > minimum) &
                               (diffs < maximum))).flatten()

    # Function to return whether a number is within epsilon of a specified value
    def within(e, a, r): return (a > e - r) & (a < e + r)
    # Apply radius around point to remove from data as well as median filtering
    remove = np.array([np.arange(i, i + radius) for i in remove if
                       any(within(np.median(means[:i]), means[i:i+radius], epsilon)) and
                       all(diffs[i-radius:i+radius] < maximum)]).flatten()
    # Ensure indices are not out of bounds
    remove = remove[(remove > 1) & (remove < len(means))]
    # Remove duplicate indices
    remove = np.unique(remove)

    # Remove spikes and return new dataframe
    return df.drop(remove)


def process_file(name, output, events):
    """
    Reads a raw data file and puts the data into a pandas dataframe.
    Then processes the data and saves the files.
    
    Arguments:
        name {str} -- a path to the data file
        output {str} -- a path to the desired output folder
        events {list} -- a list of events to mark on the diagram
    """
    global measure

    # Read data file (specific to InCytim2 software)
    dummy = False
    time = False
    data = []
    eventtimes = []
    with open(name) as lines:
        for line in lines:
            if 'Event_Times' in line:
                time = True
            elif 'Horizontal' in line:
                time = False
            elif 'DATA_AFTER_THIS_LINE' in line:
                dummy = True
            elif time:
                eventtimes.append(float(line))
            elif dummy:
                data.append(line.split())

    # Sets top row of dataframe to be the header
    filename = os.path.basename(name)[:-4]
    df = pd.DataFrame(data)
    header = df.iloc[0]
    df = df[1:]
    df.columns = header

    # Processes data
    processed_data, processed_outliers, graph_data, graph_outliers = process_data(
        df)

    # Spike detection
    graph_outliers = spike_detection(graph_outliers)

    # Saves CSV files
    processed_data.to_csv(os.path.join(output, filename + '.csv'), index=False)
    processed_outliers.to_csv(os.path.join(
        output, filename + '_outliers.csv'), index=False)

    # Saves graphs
    save_figure(graph_data, filename, output, events, eventtimes)
    save_figure(graph_outliers, filename + '_outliers',
                output, events, eventtimes)


def generate_average(names, output):
    """
    Reads previously processed CSV files and averages them together.
    Also outputs a despiked version of the data.
    
    Arguments:
        names {list} -- a list of paths to the CSV files
        output {str} -- a path to the desired output folder
    """
    global measure

    # Calcium/pH mode
    meanname = ''
    if measure == 'Calcium':
        meanname = 'Mean Calcium (nM)'
    elif measure == 'pH':
        meanname = 'Mean pH'

    # Drop empty columns and determine shortest experiment
    means = pd.DataFrame()
    ratios = pd.DataFrame()
    mintime = sys.maxsize
    for name in names:
        df = pd.read_csv(name)
        df = df.dropna(axis='columns', how='all')
        # Inserts data
        means[os.path.basename(name)[:-4] + meanname[4:]] = df[meanname]
        ratios[os.path.basename(name)[:-4] + ' Ratio'] = df['Mean Ratio']
        # Updates shortest time
        time = df['Time (s)'].iloc[-1]
        if time < mintime:
            mintime = time

    # Remove excess data and concatenate
    means = means.dropna()
    ratios = ratios.dropna()
    averages = pd.concat([means, ratios], axis=1)

    # Alternates entries from means and ratios
    cols = [None] * (len(means.columns) + len(ratios.columns))
    cols[::2] = means.columns.tolist()
    cols[1::2] = ratios.columns.tolist()
    averages = averages[cols]

    # Creates a common time axis
    timestamps = np.arange(0, mintime, 6)
    averages.insert(0, 'Time (s)', timestamps)

    # Insert empty columns
    l = len(averages.columns)
    c = 0
    for i in range(1, l + 1, 2):
        averages.insert(i + c, '', np.nan, allow_duplicates=True)
        c += 1

    # Averages data
    averages['Mean Ratio'] = ratios.mean(axis=1)
    averages[meanname] = means.mean(axis=1)

    # Spike detection
    averages_outliers = spike_detection(averages)

    # String concatenation of all filenames
    filename = ''.join(map(os.path.basename, names)
                       ).replace('.csv', ', ').strip(', ')

    # Saves CSV files
    averages.to_csv(os.path.join(output, filename + '.csv'), index=False)
    averages_outliers.to_csv(os.path.join(
        output, filename + '_despiked.csv'), index=False)

    # Saves graphs
    save_figure(averages, filename, output, [], [])
    save_figure(averages_outliers, filename + '_despiked', output, [], [])


def about():
    """About message box in file menu"""
    tk.messagebox.showinfo(
        'Information', 'This program has various data analysis tools for the NAD Lab.\n'
        + 'Made by Archie Shahidullah for the University of Arizona.\n© 2019.')


def selectfiles(entrybox):
    """
    Displays and updates selected files
    
    Arguments:
        entrybox {tk.Entry} -- a TKinter entrybox
    """
    global names

    # Prompt user to select files
    names = askopenfilenames()
    entrybox.delete(0, 'end')
    # Display list without brackets on GUI
    entrybox.insert(0, str([os.path.basename(name)
                            for name in names]).strip('[]'))


def selectoutput(entrybox):
    """
    Displays and updates output directory
    
    Arguments:
        entrybox {tk.Entry} -- a Tkinter entrybox
    """
    global output

    # Prompt user to select directory
    output = askdirectory()
    entrybox.delete(0, 'end')
    entrybox.insert(0, output)


def update_times(entrybox1, entrybox2):
    """
    Updates baseline times
    
    Arguments:
        entrybox1 {tk.Entry} -- Tkinter entrybox with initial time
        entrybox2 {tk.Entry} -- Tkinter entrybox with final time
    """
    global itime, ftime

    itime = int(entrybox1.get())
    ftime = int(entrybox2.get())


def update_base(entrybox1, entrybox2):
    """
    Updates thresholds for "bandpass" filter
    
    Arguments:
        entrybox1 {tk.Entry} -- Tkinter entrybox with lower threshold
        entrybox2 {tk.Entry} -- Tkinter entrybox with upper threshold
    """
    global lbase, ubase

    lbase = float(entrybox1.get())
    ubase = float(entrybox2.get())


def update_events(entrybox):
    """
    Updates user-defined drug names
    
    Arguments:
        entrybox {tk.Entry} -- a Tkinter entrybox
    """
    global events

    # Extract event names from entrybox
    events = entrybox.get().replace(' ', '').split(',')


def process(execute):
    """
    Processes selected files
    
    Arguments:
        execute {bool} -- a bool that determines whether all conditions 
                          for processing have been met
    """
    global names, output, measure

    # Check if conditions have been met
    if not execute:
        tk.messagebox.showinfo('Error!', 'Please Complete All Steps.')
        return

    # Process data and display confirmation
    for name in names:
        process_file(name, output, events)
    tk.messagebox.showinfo('Completed', 'Task Successful!')


def average(execute):
    """
    Averages selected files
    
    Arguments:
        execute {bool} -- a bool that determines whether all conditions 
                          for processing have been met
    """
    global names, output

    # Check if conditions have been met
    if not execute:
        tk.messagebox.showinfo('Error!', 'Please Complete All Steps.')
        return

    # Average data and display confirmation
    generate_average(names, output)
    tk.messagebox.showinfo('Completed', 'Task Successful!')


def set_meas(e1, e2, s):
    """
    Updates Calcium/pH mode
    
    Arguments:
        e1 {tk.Entry} -- Tkinter entrybox with lower threshold
        e2 {tk.Entry} -- Tkinter entrybox with upper threshold
        s {str} -- Calcium/pH mode
    """
    global lbase, ubase, measure

    # Set default thresholds based on mode
    if not e1 is None:
        if s == 'Calcium':
            lbase = 50
            ubase = 150
        elif s == 'pH':
            lbase = 6.7
            ubase = 7.3

        # Update thresholds on GUI
        e1.delete(0, 'end')
        e1.insert(0, str(lbase))
        e2.delete(0, 'end')
        e2.insert(0, str(ubase))

    # Update mode globally
    measure = s


if __name__ == '__main__':
    # Creates window
    root = tk.Tk()
    root.title('NAD Lab Tools')

    # Creates menu
    menubar = tk.Menu(root)

    filemenu = tk.Menu(menubar, tearoff=0)
    filemenu.add_command(label='Exit', command=root.destroy)
    menubar.add_cascade(label='File', menu=filemenu)

    helpmenu = tk.Menu(menubar, tearoff=0)
    helpmenu.add_command(label='About', command=about)
    menubar.add_cascade(label='Help', menu=helpmenu)

    # Adds menu
    root.config(menu=menubar)

    # Creates notebook for multiple tabs
    nb = ttk.Notebook(root)

    # Data analysis tab
    page1 = ttk.Frame(nb)

    # Steps for processing data
    stepone = tk.Label(page1, text='Step One: Select Data Files')
    steptwo = tk.Label(page1, text='Step Two: Select Output Folder')
    stepthree = tk.Label(page1, text='Step Three: Select Baseline Times')
    stepfour = tk.Label(page1, text='Step Four: Select Calcium or pH')
    stepfive = tk.Label(page1, text='Step Five: Select Outlier Range')
    stepsix = tk.Label(page1, text='Step Six: Label Drugs')
    stepseven = tk.Label(page1, text='Step Seven: Process Data!')

    # Frames embedded in tab
    frame1 = tk.Frame(page1)
    frame2 = tk.Frame(page1)
    frame3 = tk.Frame(page1)

    # Entry boxes for parameters
    entry1 = tk.Entry(page1)
    entry2 = tk.Entry(page1)
    entry_itime = tk.Entry(frame1, width=5)
    entry_itime.insert(0, str(itime))
    entry_ftime = tk.Entry(frame1, width=5)
    entry_ftime.insert(0, str(ftime))
    entry_lbase = tk.Entry(frame2, width=5)
    entry_lbase.insert(0, str(lbase))
    entry_ubase = tk.Entry(frame2, width=5)
    entry_ubase.insert(0, str(ubase))
    entry_events = tk.Entry(page1)
    entry_events.insert(0, 'Drug 1, Drug 2')

    # Buttons to control analysis
    filebutton = tk.Button(page1, text='Select files...',
                           padx=5, pady=5, command=lambda: selectfiles(entry1))
    outputbutton = tk.Button(page1, text='Select folder...',
                             padx=5, pady=5, command=lambda: selectoutput(entry2))
    timebutton = tk.Button(page1, text='Set Baseline Times',
                           padx=5, pady=5, command=lambda: update_times(entry_itime, entry_ftime))
    calcium_button = tk.Button(frame3, text='Calcium',
                               padx=5, pady=5, command=lambda: set_meas(entry_lbase, entry_ubase, 'Calcium'))
    pH_button = tk.Button(frame3, text='pH',
                          padx=5, pady=5, command=lambda: set_meas(entry_lbase, entry_ubase, 'pH'))
    basebutton = tk.Button(page1, text='Set Outlier Range',
                           padx=5, pady=5, command=lambda: update_base(entry_lbase, entry_ubase))
    eventbutton = tk.Button(page1, text='Set Drug Names',
                            padx=5, pady=5, command=lambda: update_events(entry_events))
    processbutton = tk.Button(page1, text='Process!', padx=5, pady=5, command=lambda: process(
                              len(entry1.get()) != 0 and len(entry2.get()) != 0))

    #####################
    ## Begin Placement ##
    #####################

    stepone.grid(row=0, column=0, padx=10, pady=10)
    entry1.grid(row=1, column=0, padx=10, pady=10)
    filebutton.grid(row=1, column=1, padx=10, pady=10)

    steptwo.grid(row=2, column=0, padx=10, pady=10)
    entry2.grid(row=3, column=0, padx=10, pady=10)
    outputbutton.grid(row=3, column=1, padx=10, pady=10)

    stepthree.grid(row=4, column=0, padx=10, pady=10)
    frame1.grid(row=5, column=0, padx=10, pady=10)
    entry_itime.grid(row=0, column=0, padx=10, pady=10)
    entry_ftime.grid(row=0, column=1, padx=10, pady=10)
    timebutton.grid(row=5, column=1, padx=10, pady=10)

    stepfour.grid(row=0, column=2, padx=10, pady=10)
    frame3.grid(row=0, column=3, padx=10, pady=10)
    calcium_button.grid(row=0, column=0, padx=10, pady=10)
    pH_button.grid(row=0, column=1, padx=10, pady=10)

    stepfive.grid(row=1, column=2, padx=10, pady=10)
    frame2.grid(row=2, column=2, padx=10, pady=10)
    entry_lbase.grid(row=0, column=0, padx=10, pady=10)
    entry_ubase.grid(row=0, column=1, padx=10, pady=10)
    basebutton.grid(row=2, column=3, padx=10, pady=10)

    stepsix.grid(row=3, column=2, padx=10, pady=10)
    entry_events.grid(row=4, column=2, padx=10, pady=10)
    eventbutton.grid(row=4, column=3, padx=10, pady=10)

    stepseven.grid(row=5, column=2, padx=10, pady=10)
    processbutton.grid(row=5, column=3, padx=10, pady=10)

    #####################
    ##  End Placement  ##
    #####################

    # Tab for averaging
    page2 = ttk.Frame(nb)

    # Embedded frame
    frame4 = tk.Frame(page2)

    # Steps
    sstepone = tk.Label(page2, text='Step One: Select Spreadsheets')
    ssteptwo = tk.Label(page2, text='Step Two: Select Output Folder')
    sstepthree = tk.Label(page2, text='Step Three: Select Calcium or pH')
    sstepfour = tk.Label(page2, text='Step Four: Generate Averages!')

    # Entry fields
    entry_files = tk.Entry(page2)
    entry_output = tk.Entry(page2)

    # Buttons
    files_button = tk.Button(page2, text='Select files...',
                             padx=5, pady=5, command=lambda: selectfiles(entry_files))
    output_button = tk.Button(page2, text='Select folder...',
                              padx=5, pady=5, command=lambda: selectoutput(entry_output))
    calcium_button2 = tk.Button(frame4, text='Calcium',
                                padx=5, pady=5, command=lambda: set_meas(None, None, 'Calcium'))
    pH_button2 = tk.Button(frame4, text='pH',
                           padx=5, pady=5, command=lambda: set_meas(None, None, 'pH'))
    average_button = tk.Button(page2, text='Average!', padx=5, pady=5, command=lambda: average(
                               len(entry_files.get()) != 0 and len(entry_output.get()) != 0))

    #####################
    ## Begin Placement ##
    #####################

    sstepone.grid(row=0, column=0, padx=10, pady=10)
    entry_files.grid(row=1, column=0, padx=10, pady=10)
    files_button.grid(row=1, column=1, padx=10, pady=10)

    ssteptwo.grid(row=2, column=0, padx=10, pady=10)
    entry_output.grid(row=3, column=0, padx=10, pady=10)
    output_button.grid(row=3, column=1, padx=10, pady=10)

    sstepthree.grid(row=4, column=0, padx=10, pady=10)
    frame4.grid(row=4, column=1, padx=10, pady=10)
    calcium_button2.grid(row=0, column=0, padx=10, pady=10)
    pH_button2.grid(row=0, column=1, padx=10, pady=10)

    sstepfour.grid(row=5, column=0, padx=10, pady=10)
    average_button.grid(row=5, column=1, padx=10, pady=10)

    #####################
    ##  End Placement  ##
    #####################

    # Add all tabs to window
    nb.add(page1, text='Data Analysis')
    nb.add(page2, text='Average Data')
    nb.pack(expand=1, fill='both')

    # Begin Tkinter main loop
    root.mainloop()
