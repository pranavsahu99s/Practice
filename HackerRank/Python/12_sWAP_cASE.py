# question
'''
You are given a string and your task is to swap cases. In other words, convert all lowercase letters to uppercase letters and vice versa.

For Example:
Www.HackerRank.com > wWW.hACKERrANK.COM
Pythonist 2 > pYTHONIST 2

Function Description
Complete the swap_case function in the editor below.

swap_case has the following parameters:
. string s: the string to modify

Returns
. string: the modified string

Input Format
A single line containing a string s.

Constraints
0<len(s) ≤ 1000

Sample Input O
HackerRank.com presents "Pythonist 2".

Sample Output O
hACKERrANK.COM PRESENTS "pYTHONIST 2".
'''
# solution
def swap_case(s):
    new_s = ''
    for i in s:
        if i.isupper():
            i = i.lower()
        elif i.islower():
            i = i.upper()
        else:
            new_s += i
            continue
        new_s += i
    return new_s

if __name__ == '__main__':
    s = input()
    result = swap_case(s)
    print(result)
