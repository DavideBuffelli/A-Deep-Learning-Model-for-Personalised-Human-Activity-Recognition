from test01 import perform_test01
from test02 import perform_test02
from test03 import perform_test03

# This script simply performs all the tests specified in the tests_to_do list.
tests_to_do = ["01", "02", "03"]

for test in tests_to_do:
	print("Start Test ", test)
	eval("perform_test" + test + "()")
	print("End Test ", test)