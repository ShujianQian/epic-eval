## Gacco

| use atomic  | num warehouse | txn type  | submit time | initialization time | execution time |
|:-----------:|:-------------:|:---------:|------------:|--------------------:|---------------:|
|    lock     |       1       | new order |       344us |               184us |      124'592us |
|    lock     |      16       | new order |       355us |               224us |        5'297us |
|    lock     |       1       |  payment  |       164us |               225us |   54'756'309us |
|    lock     |      16       |  payment  |       153us |               214us |    1'150'156us |
| commutative |       1       | new order |       143us |                 2us |        1'477us |
| commutative |      16       | new order |       149us |                 2us |        1'050us |
| commutative |       1       |  payment  |       124us |                 1us |          141us |
| commutative |      16       |  payment  |       136us |                 2us |           83us |
|             |               |           |             |                     |                |    
|    false    |       1       | new order |       700us |               937us |  131'436'012us |
|    false    |      16       | new order |       696us |               934us |    1'098'339us |

## Epic

| use atomic  | num warehouse | txn type  | submit time | initialization time | execution time |
|-------------|:-------------:|:---------:|------------:|--------------------:|---------------:|
|             |       1       | new order |       909us |              1212us |       21'804us |
|             |      16       | new order |       914us |              1123us |        2'003us |
|             |       1       |  payment  |       157us |               195us |      100'114us |
|             |      16       |  payment  |       176us |               221us |        7'160us |
|             |       1       | mix (1:1) |       430us |               713us |       49'842us |
|             |      16       | mix (1:1) |       454us |               691us |        4'170us |
