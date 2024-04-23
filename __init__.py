from logging import PercentStyle
from .pyfixer import PyFix

df = PyFix.df # pd.DataFrame(x)
dfx = PyFix.dfx # pd.DataFrame(x)

vc = PyFix.vc
vcdf = PyFix.vcdf 
vcdf_all = PyFix.vcdf_all 
vcdf7 = PyFix.vcdf7 # df[{column}].value_counts().reset_index().head(10)
vcdf10 = PyFix.vcdf10 # df[{column}].value_counts().reset_index().head(10)
vcdf20 = PyFix.vcdf20 # df[{column}].value_counts().reset_index().head(20)
svd = PyFix.svd

snake_df = PyFix.snake_df
snake = PyFix.snake
unsnake = PyFix.snake

pl = PyFix.pl # print(len(x))
pls = PyFix.pls # print({text}, len(x))
rdup = PyFix.rdup

any_in = PyFix.any_in
all_in = PyFix.all_in
not_in = PyFix.not_in
to_int = PyFix.to_int

breakdown = PyFix.breakdown
bd = PyFix.breakdown
hist = PyFix.hist
hist7 = PyFix.hist7
hist10 = PyFix.hist10
hist20 = PyFix.hist20

bucket = PyFix.bucket
perc = PyFix.perc
