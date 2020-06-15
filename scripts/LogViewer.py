import tkinter
import tkinter.font as tkFont
from tkinter import ttk
from tkinter import Tk
from tkinter.filedialog import askopenfilename
from tkinter import *
import csv
import os

def get_file_path():
    root = Tk()
    root.withdraw()
    file_path = askopenfilename()
    root.destroy()
    return file_path

def read_csv():
    file_path = get_file_path()
    csv.register_dialect('myDialect', delimiter = ';', skipinitialspace=True)

    # read the data from file
    with open(file_path, 'r') as csv_file:
        reader = csv.reader(csv_file, dialect='myDialect')

        timestamp = []
        log_type = []
        file_path = []
        line_number = []
        description = []

        # column numbers in csv file with data to be read out
        column_timestamp = 0
        column_log_type = 1
        column_file_path = 2
        column_line_number = 3
        column_description = 4

        first_line = True

        for row in reader:
            if first_line==True:
                first_line = False
                continue

            # remove square brackets at log-type string
            current_log_type =  str(row[column_log_type])
            open_brace_idx = current_log_type.find('[')
            close_brace_idx = current_log_type.find(']')
            current_log_type = current_log_type[open_brace_idx+1:close_brace_idx]

            timestamp.append(int(row[column_timestamp]))
            log_type.append(current_log_type)
            file_path.append(row[column_file_path])
            line_number.append(row[column_line_number])
            description.append(row[column_description])

    csv_file.close()

    return zip(*(timestamp, log_type, file_path, line_number, description))

def move_data_to_head(tree, data):
    for index, item in enumerate(data):
        tree.move(item[1], '', index)

def sort_by(tree, colomn, descending):
    # grab values to sort
    data = [(tree.set(child, colomn), child) for child in tree.get_children('')]

    # sort data as the first value of data except timestamp
    if (colomn != "Timestamp"):
        # sort data as the first value of data
        data.sort(reverse=descending)

    # move the data as the result of sorting
    move_data_to_head(tree, data)

    # switch the heading so that it will sort in the opposite direction
    tree.heading(colomn, command=lambda colomn=colomn: sort_by(tree, colomn, int(not descending)))

def search(entry_search, tree):
    search_info = entry_search.get()

    for colomn in tree_columns:
        data = [(tree.set(child, colomn), child) for child in tree.get_children('')]
        found_data = []
        for line in range(len(data)):
            result = data[line][0].find(search_info)
            # if not found, result will be -1
            if result != -1:
                found_data.append(data[line])
        # move the found data to the head
        move_data_to_head(tree, found_data)

def set_up_container_for_search(self):
    container_search = ttk.Frame()
    container_search.pack(side=TOP, anchor=NE)
    entry_search = Entry(container_search)
    entry_search.pack(side=LEFT, fill=X, expand=YES)
    # connect the keyboard "Return" to the search function
    entry_search.bind("<Return>", lambda event: search(entry_search, self.tree))
    Button(container_search, text='Search', command=lambda: search(entry_search, self.tree)).pack(side=LEFT, fill=X, expand=YES)

def set_up_container_for_log_viewer(self):
    container_log_view = ttk.Frame()
    container_log_view.pack(fill='both', expand=True)
    self.tree = ttk.Treeview(columns=tree_columns, show="headings")
    vertical_scrollbar = ttk.Scrollbar(orient="vertical", command=self.tree.yview)
    horizen_scrollbar = ttk.Scrollbar(orient="horizontal", command=self.tree.xview)
    self.tree.configure(yscrollcommand=vertical_scrollbar.set, xscrollcommand=horizen_scrollbar.set)
    self.tree.grid(column=0, row=0, sticky='nsew', in_=container_log_view)
    vertical_scrollbar.grid(column=1, row=0, sticky='ns', in_=container_log_view)
    horizen_scrollbar.grid(column=0, row=1, sticky='ew', in_=container_log_view)

    container_log_view.grid_columnconfigure(0, weight=1)
    container_log_view.grid_rowconfigure(0, weight=1)

class App(object):
    def __init__(self):
        self.tree = None
        self.entry = None
        self.setup_widgets()
        self.build_tree()

    def setup_widgets(self):
        # set up the container for search
        set_up_container_for_search(self)

        # set up the container for log viewer
        set_up_container_for_log_viewer(self)

    def build_tree(self):
        for colomn in tree_columns:
            self.tree.heading(colomn, text=colomn.title(),
                command=lambda c=colomn: sort_by(self.tree, c, 0))

            self.tree.column(colomn, width=tkinter.font.Font().measure(colomn.title()))

        for item in tree_data:
            self.tree.insert('', 'end', values=item)

            # adjust columns length if necessary
            for indx, val in enumerate(item):
                ilen = tkinter.font.Font().measure(val)
                if self.tree.column(tree_columns[indx], width=None) < ilen:
                    self.tree.column(tree_columns[indx], width=ilen)

tree_columns = ("Timestamp", "LogType", "FilePath", "LineNumber", "LogMessage")
tree_data = read_csv()

def main():
    root = tkinter.Tk()
    root.wm_title("Log Viewer")
    root.wm_iconname("mclist")
    app = App()
    root.mainloop()

if __name__ == "__main__":
    main()
