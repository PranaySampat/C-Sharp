using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Numerics;
using System.Threading.Tasks;

namespace Assignment1
{
	class Program
	{
		

		static void Main(string[] args)
		{
			/*
			Console.WriteLine("Here the list of 10 Primonacci, find prime of fibonacci");
			var primonacci = Unfold3(1, (a, b) => a + b);
			foreach(var x in primonacci.Take(20))
			{
				Console.WriteLine(x);
			}
			Console.WriteLine("Even after using long, the program crushesh after 11th fibonacci and starts printing negative" +
			                  "number, so now i am printing fibonacc of prime numbers");
			                  */
			Console.WriteLine("Fibonacci of primes, primonacci");
			//LINQ expression that computes primenumber and stores as the list
			IEnumerable<int> primes =
				Enumerable.Range(2, Int32.MaxValue - 1).
				          Where(number => Enumerable.Range(2, (int)Math.Sqrt(number) - 1).
				                All(divisor => number % divisor != 0));

			IEnumerable<int> primonacci2 = Primonacci(primes, (a, b) => a + b);
			Console.WriteLine("Printing first 10 primenoacci");
			foreach(var primeNacci in primonacci2.Take(10)) {
				Console.WriteLine(primeNacci);
			}
			Console.WriteLine("Printing 20 primenacci number after 1000");
			foreach (var primeNacciAfter1000 in primonacci2.Skip(1000).Take(20))
			{
				Console.WriteLine(primeNacciAfter1000);
			}
			Console.WriteLine("Printing 10001 primonacci");
			foreach (var primoNacci10001 in primonacci2.Skip(10000).Take(1)){
				Console.WriteLine(primoNacci10001);
			}
			Console.WriteLine("Printing 100 Prime Numbers");
			primes.Take(100).ToList().ForEach(prime => Console.WriteLine(prime));


		}

		//yeild function for primonacci
        private static IEnumerable<T> Primonacci<T>(IEnumerable<T> primeNums, Func<T, T, T> accumulator)
		{
			T c;
			int i = 0;
			while (i >= 0)
			{
				c = accumulator(primeNums.ElementAt(i), primeNums.ElementAt(i + 1));
				yield return c;
				i++;
			}
		}



		private static IEnumerable<T> Unfold3<T>(T seed, Func<T, T, T> accumulator)
		{
			var a = seed;
			var b = seed;
			T c;
			while (true)
			{
				long number= (Convert.ToInt64(b));


				Int64 end = (Int64)(Math.Sqrt(number) + 1);
				Boolean primeCase = false;
				if (number == 2)
				{
					yield return b;
				}
				else if (number != 1 && number % 2 != 0)
				{

					for (int i = 2; i <= end; i++)
					{
						if (number % i == 0)
						{
							primeCase = true;
							break;
						}
					}
					if (!primeCase)
					{
						yield return b;
					}

				}


				c = b;
				b = accumulator(a, b);
				a = c;
			}

		}
		/*
		 static BigInteger sqrt(BigInteger n)
		{
			BigInteger a = BigInteger.One;
			BigInteger b = new BigInteger(n.shiftRight(5).add(new BigInteger("8")).toString());
			while (b.CompareTo(a) >= 0)
			{
				BigInteger mid = new BigInteger(a.add(b).shiftRight(1).toString());
				if (mid.multiply(mid).compareTo(n) > 0) b = mid.subtract(BigInteger.ONE);
				else a = mid.add(BigInteger.One);
			}
			return a.subtract(BigInteger.One);
		}

*/
	}
	}


