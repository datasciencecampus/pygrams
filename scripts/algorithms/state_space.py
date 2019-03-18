# import csv
# import os
# import rpy2.robjects as robjects
#
#
# class StateSpaceModel(object):
#
#     def __init__(self, data_in, num_prediction_periods):
#         if not all(isinstance(x, float) for x in data_in):
#             raise ValueError('Time series must be all float values')
#
#         self.__history = data_in
#         self.__num_prediction_periods = num_prediction_periods
#
#     @property
#     def configuration(self):
#         return None
#
#     def predict_counts(self):
#         cwd = os.getcwd()
#         rwd = robjects.r('getwd()')
#
#         if '/scripts/algorithms/code' not in rwd[0]:
#             wd = '''setwd(''' + '\'' + cwd + '''/scripts/algorithms/code')'''
#             robjects.r(wd)
#         rwd = robjects.r('getwd()')
#         output_path = rwd[0] + '/buffer.csv'
#         print("path: "+output_path)
#
#         with open(output_path, "w") as file:
#             writer = csv.writer(file, delimiter='\n')
#             writer.writerow(self.__history)
#
#         robjects.r('''
#                source('predict')
#         ''')
#
#         r_func = robjects.globalenv['predict']
#         out = r_func("buffer.csv",self.__num_prediction_periods)
#         os.chdir(cwd)
#         return out[0]
