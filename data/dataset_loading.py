from typing import List


def _do_assertions(list_of_bools: List[bool], list_of_ints: List[int], list_of_strs: List[str]):
    # TODO
    for bool_ in list_of_bools:
        assert isinstance(
            bool_, bool), f"input parameters have to be bool, {bool_} is not"
    for int_ in list_of_ints:
        assert isinstance(
            int_, int), f"input parameters have to be int, {int_} is not"
    for str_ in list_of_strs:
        assert isinstance(
            str_, str), f"input parameters have to be str, {str_} is not"
