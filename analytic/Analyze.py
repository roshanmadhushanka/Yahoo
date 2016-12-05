def categorical(pandas_frame, column_name):
    categories = {}
    column_data = pandas_frame[column_name]

    for data in column_data:
        if categories.has_key(data):
            categories[data] += 1
        else:
            categories[data] = 1

    print "Instances :", len(column_data)
    print "Category Summary"
    print "----------------"
    for key in categories.keys():
        print key, ":", categories[key]

