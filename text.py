text = "abcd"
shift = 3

key = "abcdefghijklmnopqrstuvwxyz"
 
def encrypt(text,shift):
	
	result = ''
	for char in text.lower():
		i = (key.index(char) + shift) %26
		result+= key[i]
	return result

	
text2 = encrypt(text,shift)
print ("encrypted code is ", text2)

def decrypt(text2,shift):
	result1= ''
	
	for char in text2:

		i = (key.index(char) - shift) %26
		result1 +=key[i]
	print ("decrypted code is ", result1)


decrypt(text2,shift)