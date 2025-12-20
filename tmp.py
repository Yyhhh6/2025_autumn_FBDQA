# follow(user1, user2):->
# def main(input_list: list):
#     output = None
#     skip_list = []
#     for user1 in res_list:
#         output = user1
#         for user2 in input_list:
#             if user1 != user2:
#                 if follow(user1, user2):
#                     output = None
#                     break
                
#                 if not follow(user2, user1):
#                     output = None
#                     break
#                 else:
#                     res_list.remove(user2)
#         if output is not None:
#             return output
#     return None

def main(input_list: list):
    current_list = input_list.copy()
    if current_list is not None:
        current_list_ = []
        for i in range(len(current_list),step=2):
            user1 = input_list[i]
            user2 = input_list[i+1]
            if follow(user1, user2):
                current_list_.append(user2)
            else:
                current_list_.append(user1)
        current_list = current_list_
    
