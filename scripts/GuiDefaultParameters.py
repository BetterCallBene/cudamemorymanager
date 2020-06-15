import tkinter as tk
import re, sys, os
from tkinter import filedialog
from tkinter import messagebox
from shutil import copyfile
import pandas as pd

root = tk.Tk()
root.title("Parameter Applicator")

# Output color configuration
Color_RED = '\033[91m'
Color_BLUE = '\033[94m'
Color_END = '\033[0m'

# Variable for gui layout
Field_Distance = 6
Label_Size = 20

# Path of data file
current_dir = os.path.dirname(os.path.realpath(__file__))
root_dir = os.path.dirname(current_dir)
data_path = root_dir + "\params\parameters.json"

# Constants
Error_Not_Found = "NOT_FOUND"
Not_Changeable = "NOT_CHANGEABLE"
bool_dict = {
    "true":1,
    "True":1,
    "false":0,
    "False":0
    }
string_for_float32 = "float32"


def write_parameter_to_json(parameter_name, parameter_value):
    data = {
        'Name':[parameter_name],
        'Value':[parameter_value]
        }
    data_frame = pd.DataFrame(data)
    if os.path.isfile(data_path):
        read_data_frame = (pd.read_json(data_path, 
            orient='records')).astype('str', errors='ignore')
        read_data_frame.columns=['Name','Value']
        read_data_frame_analyzed = read_data_frame['Name'] == parameter_name
        entry_exists = (read_data_frame_analyzed.any())
        if not entry_exists:
            new_data_frame = read_data_frame.append(data_frame,
                ignore_index = True)
        else:
            index_of_existing_entry = read_data_frame_analyzed.index[
                read_data_frame_analyzed.values == True]
            new_data_frame = read_data_frame
            new_data_frame.at[index_of_existing_entry, 'Value'] = \
                (str)(parameter_value)
        new_data_frame.to_json(data_path, orient='records')
    else:
        data_frame.to_json(data_path, orient='records')


class MyToolTip(tk.Toplevel):
    TIP_X_OFFSET = 8
    TIP_Y_OFFSET = 8
    AUTO_CLEAR_TIME = 1000 # Millisek. (hier is 1 sek)

    def __init__(self, xpos, ypos, message, auto_clear=False):

        self.xpos = xpos
        self.ypos = ypos
        self.message = message
        self.auto_clear = auto_clear

        tk.Toplevel.__init__(self)
        self.overrideredirect(True)
        self.labelframe = tk.LabelFrame(self)
        self.labelframe.pack(fill="both", expand="yes")
        self.label = tk.Label(self.labelframe, compound='left',
            text=self.message)
        self.label.pack()

        self.geometry("+%d+%d" % (self.xpos+self.TIP_X_OFFSET,
            self.ypos+self.TIP_Y_OFFSET))

        if self.auto_clear:
            self.after(self.AUTO_CLEAR_TIME, self.clear_tip)

    def clear_tip(self):
        self.destroy()


def entry_mouse_enter(event):
    output_string = ""
    if event.widget.cget('text') == "Write to .json File": 
        root.my_tool_tip = MyToolTip(event.x_root, event.y_root,
            "The changed parameter will be written to .json file")
        return
    elif event.widget.cget('text') == "Write Parameters to .hpp file":
        root.my_tool_tip = MyToolTip(event.x_root, event.y_root,
            "The changed parameters will be set as dafault parameter" + 
            "and written to .hpp")
        return

    parameter = event.widget.cget("text")
    range_min = get_min(parameter)
    range_max = get_max(parameter)

    output_string = "Datatyp: " + parameter_datatype_ditc.get(parameter) \
                    + "\n"

    if  range_min != Error_Not_Found and  range_max != Error_Not_Found:
        output_string = output_string + "Range: [" + range_min + "," + \
            range_max +"]"
        if parameter in parameter_description_dict:
            output_string = output_string + "\n" + "Description: " + \
                parameter_description_dict.get(parameter)     
    root.my_tool_tip = MyToolTip(event.x_root, event.y_root, output_string)


def entry_mouse_leave(event):
    root.my_tool_tip.destroy()


def show_warning_out_of_range(parameter_out_of_range_dict):
    warning_string = "Following parameters are out of range," +\
        " please reset them \n"
    parameter_string = buildErrorStringForParameters(
        parameter_out_of_range_dict)
    messagebox.showerror("Error", warning_string + parameter_string)


def buildErrorStringForParameters(parameter_out_of_range_dict):
    parameter_string = " "
    for key in parameter_out_of_range_dict:
        parameter_string = parameter_string + "Parameter:" + key + "=" + \
                           parameter_out_of_range_dict.get(key) + \
                           " is out of the range [" + get_min(key) + \
                           ", " + get_max(key) + "]\n"
    return parameter_string


def get_min(parameter):
    if parameter in parameter_range_dict:
        return (parameter_range_dict.get(parameter))[0]
    else:
        return Error_Not_Found


def get_max(parameter):
    if parameter in parameter_range_dict:
        return (parameter_range_dict.get(parameter))[1]
    else:
        return Error_Not_Found


def set_new_value_and_write_to_json():
    for entry in entries:
            parameter = entry[0]
            current_value = entry[1].get()
            if current_value.split("f")[0] != parameters_dict[
                    parameter].split("f")[0]:
                changed_parameters_dict[parameter] = current_value
                write_parameter_to_json(parameter, current_value)


def change_parameter_and_write_to_json():
    if check_range_of_parameter() == True:
        if messagebox.askokcancel("Info", "If you press OK, the " + 
                "changed parameters will be written to .json file"):
            set_new_value_and_write_to_json()
        activateButton(cpp_button)


def write_parameter_to_sourcefile():
    if messagebox.askokcancel("Warning", "The changed parameters " + 
            "will be written to .hpp, is this OK?"):
        buffer_file_path = os.path.abspath(
            os.path.dirname(sys.argv[0])) + "\\ParameterStorage_Buffer.hpp"
        copyfile(path, buffer_file_path)
        buffer_file = open_file(buffer_file_path, "read")
        file = open_file(path,"write")
        lines = buffer_file.readlines()
        write_lines_to_file (lines, file)
        file.close()
        buffer_file.close()
        os.remove(buffer_file_path)
    deactivate_button(cpp_button)
    initialize_entry_color()


def write_lines_to_file (lines, file):
    for i in range(0, len(lines)):
        line = lines[i]
        for parameter in list(changed_parameters_dict.keys()):
            if line.find(parameter) != -1:
                if -1 == str(changed_parameters_dict.get(parameter)
                        ).find("f") and parameter_datatype_ditc.get(parameter) \
                        == string_for_float32:
                    lines[i+1] = lines[i+1].replace(
                        str(parameters_dict.get(parameter)),
                        str(changed_parameters_dict.get(parameter)) + "f", 1)
                else:
                    lines[i+1] = lines[i+1].replace(
                        str(parameters_dict.get(parameter)),
                        str(changed_parameters_dict.get(parameter)), 1)
                parameters_dict[parameter] = changed_parameters_dict.get(parameter)
                del changed_parameters_dict[parameter]
                break
        file.write(line)


def open_file(file_path, read_or_write):
    if read_or_write == "read":
        try:
            file = open(file_path,"r")
        except IOError:
            print(Color_RED + "Error:", "the file of", file_path, \
                  "can not be opened for reading" + Color_END)
        return file
    else:
        try:
            file = open(file_path, "w+")
        except IOError:
            print(Color_RED + "Error:", "the file of", file_path, \
                  "can not be opened for writing" + Color_END)
        return file


def read_source_file(file_path):
    file = open_file(file_path,"read")
    lines = file.readlines()
    parameter = ""
    current_parameter_processesd = False
    for line in lines:
        if False == current_parameter_processesd:
            parameter = read_parameter(line)
            if parameter != Error_Not_Found:
                current_parameter_processesd = True
                continue
        else:
            data_type = read_datatype(line)
            value = read_value(line)
            range_min_value = read_min(line)
            range_max_value = read_max(line)
            mutability_state = read_mutability_state(line)
            description = read_description(line)
            current_parameter_processesd = False

            if parameter != Error_Not_Found:
                if data_type != Error_Not_Found:
                    parameter_datatype_ditc[parameter] = data_type
                if value != Error_Not_Found:
                    parameters_dict[parameter] = value
                if range_min_value != Error_Not_Found and \
                        range_max_value != Error_Not_Found:
                    parameter_range_dict[parameter] = [range_min_value,
                        range_max_value]
                if mutability_state != Error_Not_Found:
                    parameter_mutability_state_dict[parameter] = \
                        mutability_state
                if description != Error_Not_Found:
                    parameter_description_dict[parameter] = description
    file.close()


def select_source_file_path():
    return filedialog.askopenfilename(
        title = "Select local ParameterStorage.hpp",
        filetypes = (("hpp files", "*.hpp"),
        ("all files", "*.*")))

def save_parameter_in_file(file_to_copy):
    file_name = filedialog.asksaveasfilename(
        title = "Path to copy of ParameterStorage.hpp",
        filetypes = (("hpp files", "*.hpp"), ("all files", "*.*"))) + ".hpp"
    copyfile(file_to_copy,file_name)


def read_parameter(line):
    parameter=re.findall('.*"p_(.*)".*',line)
    if len(parameter) == 0:
        return Error_Not_Found
    else:
        return parameter[0]


def read_datatype(line):
    data_type=re.findall('.*TypedParameter<(.*)>\(.*',line)
    if len(data_type) == 0:
        return Error_Not_Found
    else:
        return data_type[0]


def read_value(line):
    value=re.findall('TypedParameter.*\((.*),.*,.*, MutabilityState',line)
    if len(value) == 0:
        return Error_Not_Found
    else:
        return value[0]


def read_min(line):
    min_value=re.findall('TypedParameter.*, (.*),.*, MutabilityState*',line)
    if len(min_value) == 0:
        return Error_Not_Found
    else:
        return min_value[0]


def read_max(line):
    max_value=re.findall('TypedParameter.*,.*, (.*), MutabilityState.*',line)
    if len(max_value) == 0:
        return Error_Not_Found
    else:
        return max_value[0]


def read_description(line):
    description=re.findall('.*MutabilityState::.*, "(.*)".*',line)
    if len(description) == 0:
        return Error_Not_Found
    else:
        return description[0]


def read_mutability_state(line):
    mutability_state=re.findall(
        "INIT|CHANGEABLE_ON_RUNTIME|CHANGEABLE_ON_INIT|NOT_CHANGEABLE",
        line)
    if len(mutability_state) == 0:
        return Error_Not_Found
    else:
        return mutability_state[0]


def change_text_color(entry_widget, color):
    entry_widget.config(fg= color)


def handle_key_pressed_event(event):
    change_text_color(event.widget, "red")


def convert_to_float(parameter_string):
    result = parameter_string

    if parameter_string in bool_dict:
        result = bool_dict.get(parameter_string)
        return float(result) 
    elif parameter_string.find("f") != -1:
        result = parameter_string.split("f")[0]
        return float(result)
    else:
        return float(result)


def check_range_of_parameter():
    result = True

    for entry in entries:
            parameter = entry[0]
            current_value = entry[1].get()
            if current_value != parameters_dict[parameter]:
                new_value = convert_to_float(current_value)
                range_min = convert_to_float(get_min(parameter))
                range_max = convert_to_float(get_max(parameter))
                if range_min > new_value or new_value > range_max:
                    parameter_out_of_range_dict[parameter]= current_value
                elif parameter in parameter_out_of_range_dict:
                    del parameter_out_of_range_dict[parameter]

    if len(parameter_out_of_range_dict) != 0:
        show_warning_out_of_range(parameter_out_of_range_dict)
        result = False

    return result


def initialize_entry_color():
    for entry in entries:
            change_text_color(entry[1],"black")


def check_file_exists(path):
    return os.path.isfile(path)


def create_gui():
    frame = tk.Frame(root,width = 400,height = 300)
    frame.grid(row = 0,column = 0)

    tk.Label(frame, text = "Parameters").grid(column = 0, row = 0)
    tk.Label(frame, text = "Value").grid(column = 1, row = 0)

    canvas = tk.Canvas(frame, width = 400, height = 300,
        scrollregion = (0,0,0,2000))
    canvas.grid(column = 0, row = 1, columnspan = 2)
    vbar = tk.Scrollbar(frame, orient = "vertical")
    vbar.grid(column = 2, row = 1, ipady = 125)
    vbar.config(command = canvas.yview)
    canvas.config(yscrollcommand = vbar.set)

    write_parameter_to_json_button = tk.Button(frame, text = "Write to .json File",
        command = change_parameter_and_write_to_json)
    write_parameter_to_json_button.bind('<Enter>', entry_mouse_enter)
    write_parameter_to_json_button.bind('<Leave>', entry_mouse_leave)
    write_parameter_to_json_button.grid(column = 0, row = 2)

    write_parameter_to_source_file_button = tk.Button(frame, 
        text = "Write Parameters to .hpp file", 
        command = write_parameter_to_sourcefile,
        state = 'disabled')
    write_parameter_to_source_file_button.bind('<Enter>', entry_mouse_enter)
    write_parameter_to_source_file_button.bind('<Leave>', entry_mouse_leave)
    write_parameter_to_source_file_button.grid(column = 1, row = 2)

    return [canvas, write_parameter_to_source_file_button]


def fill_entries(canvas):
    for row in range(rows):
        for column in range(columns):
            element = tk.StringVar()
            if column == 0:
                new_entry = tk.Label(canvas, text = parameter_list[row])
                new_entry.bind('<Enter>', entry_mouse_enter)
                new_entry.bind('<Leave>', entry_mouse_leave)
            else:
                if parameter_mutability_state_dict.get(parameter_list[row]) != \
                        Not_Changeable:
                    state = "normal"
                else:
                    state = "disable"
                new_entry = tk.Entry(canvas, textvariable = element, state = state)
                element.set(parameters_dict.get(parameter_list[row]))
            row_coordinate = Field_Distance + Label_Size * row
            column_coordinate = Field_Distance + field_width * column * Field_Distance
            id = canvas.create_window(column_coordinate, row_coordinate,
                                      window = new_entry, anchor = "nw")
            if column == 1:
                canvas.itemconfig(id, tags = parameter_list[row])
                new_entry.bind('<Key>', lambda event: handle_key_pressed_event(event))
                entries.append((parameter_list[row],new_entry))


def deactivate_button(button):
	button['state'] = 'disabled'


def activateButton(button):
	button['state'] = 'normal'


[canvas, cpp_button] = create_gui()
path = select_source_file_path()
if messagebox.askokcancel("Copy", "Make a copy of your selected .cpp?"):
    save_parameter_in_file(path)
parameters_dict = {} 
parameter_datatype_ditc = {} 
parameter_range_dict = {} 
parameter_description_dict = {} 
parameter_mutability_state_dict = {} 
changed_parameters_dict = {} 
parameter_out_of_range_dict = {} 
entries = [] 
read_source_file(path)
rows = len(parameters_dict)
columns = 2
specific_name = "test"
field_width = 35
parameter_list = list(parameters_dict.keys())
fill_entries(canvas)
# Configurate the length of canvas
canvas.configure(scrollregion = canvas.bbox("all")) 

root.mainloop()