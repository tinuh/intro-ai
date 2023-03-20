from colorama import Fore, Back, Style
from random import randint
print("Worldle!")
print(Back.GREEN + "Correct letters in the right position will be highlighted with a green background!" + Style.RESET_ALL)
print(Back.YELLOW + "Letters in the wrong position be highlighted with a yellow background!" + Style.RESET_ALL)

#pick a random word from the word list
index = randint(0, 2300)
with open(r"words.txt", 'r') as words:
  word = words.readlines()[index].strip()

print(word)

#global variables to keep track of letters used, and avaiable
avail = ["a", "b", "c",  "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z"]
notPresent = ""
inWord = ""

#function to check the guess and return formatted string with colors
def checkWord(word: str, guess: str) -> str:
  global avail, notPresent, inWord

  checked = ""
  for i in range(len(word)):
    # if letter is correct, add green background
    if guess[i] == word[i]:
      checked += Back.GREEN + guess[i] + Style.RESET_ALL
      if (not guess[i] in inWord):
        inWord += word[i]
    # if letter is in word, add yellow background
    elif guess[i] in word:
      checked += Back.YELLOW + guess[i] + Style.RESET_ALL
      if (not guess[i] in inWord):
        inWord += guess[i]
    # if letter is not in word, replace with underscore
    else:
      checked += "_"
      if (not guess[i] in notPresent):
        notPresent += guess[i]
        del avail[avail.index(guess[i])]

  return checked

guesses = 0
while True:
  guess = ""
  #validate guess length
  while (len(guess) != 5):
    guess = input("Enter your guess: ").lower()
    if (len(guess) != 5):
      print(Fore.RED + "Guess must be 5 letters long!" + Style.RESET_ALL)

  val = checkWord(word, guess)
  print(val)

  # print out available, not present, and in word letters
  print("Available:", "".join(avail))
  print("Not Present:", notPresent)
  print("In Word:", inWord)
  
  #increment guesses
  guesses += 1

  #check if guess is correct
  if guess == word:
    print(f"You won in {guesses} guesses!")
    break