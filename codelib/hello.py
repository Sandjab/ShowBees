"""Hello World file used to check library reachability"""

# Displayed during import.
print('Hello Import [' + __name__ + ']!')

# Displayed if main
if __name__ == '__main__':
    print('Hello Main!')

# Displayed during explicit call
def hello_world():
    s = 'Hello World!'
    print(s)
    return s