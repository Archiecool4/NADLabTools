# NAD Lab Tools
This program was written for the NAD Lab at the University of Arizona by Archie Shahidullah.
It processes intracellular calcium concentration and pH measurements (from the InCytim2 software)
as well as filters the data for outliers and spikes. 

The experiment consists of placing fluorescent-stained cells under a microscope to measure either
calcium concentration or pH. Over a period of time, solutions are added to determine the response
of the quantities.

Archie Shahidullah 2019

---

This program was made using Python, using the NumPy, Pandas, and Matplotlib packages.

The repository includes:

- NADLabTools.py - The Python source code.

Use PyInstaller to generate an executable of the program. First install PyInstaller using the command line:

`pip install pyinstaller`

Then generate the executable:

`pyinstaller --onefile NADLabTools.py`
