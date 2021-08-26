#import libraries
from virtual_agent import VAModel

va_model = VAModel()
va_model.train("faqs.csv")
print("Welcome to Kandi VA!")
while True:
    print("----------------------------")
    usr_query = input("Type your query here: ")
    if usr_query.lower() == "exit":
        va_model.free_up()
        break
    else:
        response = va_model.pred_answer(usr_query)
        print(response)
