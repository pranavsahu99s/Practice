# question
'''
Example
>>> string = "abracadabra"
>>> l = list(string)
>>> l[5] = 'k'
>>> string = ''.join(l)
>>> print string
abrackdabra

Another approach is to slice the string and join it back.

Example
>>> string = string[:5] + "k" + string[6:]
>>> print string
abrackdabra

Task
Read a given string, change the character at a given index and then print the modified string.

Function Description
Complete the mutate_string function in the editor below.

mutate_string has the following parameters:
. string string: the string to change
. int position: the index to insert the character at
. string character: the character to insert

Returns
. string: the altered string

Input Format
The first line contains a string, string.
The next line contains an integer position, the index location and a string character, separated by a space.

Sample Input
STDIN           Function
-----           --------
abracadabra     s = 'abracadabra'
5 k             position = 5, character = 'k'

Sample Output
abrackdabra
'''
# solution
def mutate_string(string, position, character):
    new_string = string[: position] + character + string[position+1 :]
    return new_string

if __name__ == '__main__':
    s = input()
    i, c = input().split()
    s_new = mutate_string(s, int(i), c)
    print(s_new)
