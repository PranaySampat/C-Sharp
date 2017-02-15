using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace StackSorting
{
    public class Program
    {
        public static void Main(string[] args)
        {
            Stack<int> stack = new Stack<int>();
            stack.Push(100);
            stack.Push(54);
            stack.Push(33);
            stack.Push(44);
            Stack<int> mayDay = new Stack<int>(stack);
                        Console.WriteLine("The Input Stack is ");
            while(mayDay.Count() != 0){
                Console.Write(mayDay
                .Pop() + " ");
            }
            Stack<int> output = new Stack<int>();
            output = sortingStack(stack);
            
            Console.WriteLine("Output Stack is ");
            while(output.Count() != 0){
                Console.Write(output.Pop() + " ");
            }
        }

        public static Stack<int> sortingStack (Stack<int> input){
            Stack<int> auxilaryStack = new Stack<int>();
            while(input.Count() != 0){
                int tmp = input.Pop();
                while(auxilaryStack.Count() != 0 && auxilaryStack.Peek() > tmp){
                    input.Push(auxilaryStack.Pop());
                }
                auxilaryStack.Push(tmp);
            }
            return auxilaryStack;
        }
    }
}
