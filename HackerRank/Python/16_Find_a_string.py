# question
'''
In this challenge, the user enters a string and a substring. You have to print the number of times that the substring occurs in the given string. 
String traversal will take place from left to right, not from right to left.
NOTE: String letters are case-sensitive.

Input Format
The first line of input contains the original string. The next line contains the substring.

Constraints
1 <= len(string) <= 200
Each character in the string is an ascii character.

Output Format
Output the integer number indicating the total number of occurrences of the substring in the original string.

Sample Input
ABCDCDC
CDC

Sample Output
2
'''
# solution
def count_substring(string, sub_string):
    c = 0
    i =0
    j = len(sub_string)
    while i < len(string):
        if string.count(sub_string, i, j) :
            c+=1
        i+=1
        j+=1
    return c

if __name__ == '__main__':
    string = input().strip()
    sub_string = input().strip()
    
    count = count_substring(string, sub_string)
    print(count)
