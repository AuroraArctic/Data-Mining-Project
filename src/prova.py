def main(*args):
    parameter_dict = {}
    for user_input in argv:
        if "=" not in user_input:
            continue
        varname = user_input.split("=")[0] #Get what's left of the '='
        varvalue = user_input.split("=")[1] #Get what's right of the '='
        parameter_dict[varname] = varvalue

    # Default values
    support=0.01
    lines=0
    days=1
    min_items = 1
    max_items = 5

    total_param = ['lines','support','days','min_items','max_items']
    if 'lines' in parameter_dict.keys():
      lines = parameter_dict['lines']

    if 'support' in parameter_dict.keys():
      support = parameter_dict['support']

    if 'days' in parameter_dict.keys():
      days = parameter_dict['days']

    if 'min_items' in parameter_dict.keys():
      min_items = parameter_dict['min_items']

    if 'max_items' in parameter_dict.keys():
      max_items = parameter_dict['max_items']
