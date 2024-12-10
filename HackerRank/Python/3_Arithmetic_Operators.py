#Question
'''
Task

The provided code stub reads two integers from STDIN, a and b. Add code to print three lines where:
1. The first line contains the sum of the two numbers.
2. The second line contains the difference of the two numbers (first - second).
3. The third line contains the product of the two numbers.

Example
a=3
b=5

Print the following:
00
-2
15

Input Format
The first line contains the first integer, a.
The second line contains the second integer, b.

Constraints
1 ≤ a ≤ 1010
1 ≤ b ≤ 1010

Output Format
Print the three lines as explained above.

Sample Input O
3
2

Sample Output O
510
'''

#Solution
if __name__ == '__main__':
    a = int(input())
    b = int(input())
    
print(f"{a+b}\n{a-b}\n{a*b}") 
