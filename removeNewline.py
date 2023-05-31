import sys
# str = sys.stdin.readlines()
# str = input("Enter Sentence\n")
def remove_newline():
    out = ''
    writing = open("./texts/directFromPDFWithoutNewline.txt", "w")
    with open('./texts/directFromPDF.txt') as f:
        lines = f.readlines()

    for line in lines:
        out = out+line.rstrip()+' '
    writing.write(out)
    writing.close()
