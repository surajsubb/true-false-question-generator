import fs from "fs";
// var toJSON = require('plain-text-data-to-json')
import wd from "word-definition";

import { capitalized } from "is-verb";

// function isVerb(firstWord) {
//     //returns 1 if verb else 0 if not 
//     var ans = 0;
//     const x = wd.getDef(firstWord, "en", null, function (definition) {
//         // console.log(definition.category);
//         if (definition.category == "verb") {
//             console.log("hi");
//             ans = 1;
//         }
//     });

//     return ans;
// }
function hello() {
    console.log("hello world")
}
async function test() {


    var title = "directFromPDFWithoutNewline";
    var doc = fs.readFileSync("./texts/" + title + ".txt", 'utf8')

    var sentenceArray = [];

    var doc1 = ".";

    doc1 += doc;

    doc1 = doc1.replace(/ *\([^)]*\) */g, ""); //remove brackets
    doc1 = doc1.replace(/2022-23/g, "");
    doc1 = doc1.replace(/Fig/g, "");
    // doc1 = doc1.replace(/Figure/g, "");
    doc1 = doc1.replace(/[A-Z][A-Z]+/g, "");



    // console.log(doc1);


    for (let i = 0; i < doc1.length; i++) {

        var curS = "";
        var sentenceLetterCount = 0;

        //remove nos from the start
        //3 cases - with colon, with space or with comma

        if (doc1[i] === ".") {
            i++;
            sentenceLetterCount++;
            var startSentence = 1;

            while (i < doc1.length && doc1[i] != ".") {
                if (sentenceLetterCount < 2 && Number(doc1[i])) {
                    i++;
                    //deal w double digits later
                    if (i < doc1.length && (i == ":" || i == ",")) {
                        i++;
                    }

                } else {
                    curS += doc1[i];
                }
                i++;
                sentenceLetterCount++;
            }

            //remove a no at the end - bullet headings
            if (curS.length > 0 && Number(curS[curS.length - 1])) {
                //deal with double digit nos later
                curS = curS.slice(0, -1);
                if (curS.length > 0 && Number(curS[curS.length - 1])) {
                    curS = curS.slice(0, -1);
                }
            }

            //filter for questions
            if (curS.includes("?")) {
                curS = "";
            }

            if (curS.includes("Activity")) {
                curS = "";
            }

            var pronounsArr = ["he", "she", "his", "they", "us"]; //remove "we", "our"?

            var randomFilterArray = ["let", "discussed", "chapter"];

            //filter for pronouns
            for (var q = 0; q < pronounsArr.length; q++) {
                var temp = [];
                var x = "";
                x += " ";
                x += pronounsArr[q];
                x += " ";
                temp.push(x);

                // var x = "";
                // x += pronounsArr[q];
                // x += " ";
                // temp.push(x);

                var x = " ";
                x += pronounsArr[q];
                x += ".";
                temp.push(x);

                for (var z = 0; z < temp.length; z++) {
                    if (curS.toLowerCase().includes(temp[z])) {
                        // console.log(curS.toLowerCase());
                        // console.log(pronounsArr[q]);
                        curS = "";
                    }
                }
            }

            for (var q = 0; q < randomFilterArray.length; q++) {
                var temp = [];
                var x = "";
                x += " ";
                x += randomFilterArray[q];
                x += " ";
                temp.push(x);

                // var x = "";
                // x += randomFilterArray[q];
                // x += " ";
                // temp.push(x);

                var x = " ";
                x += randomFilterArray[q];
                x += ".";
                temp.push(x);

                for (var z = 0; z < temp.length; z++) {
                    if (curS.toLowerCase().includes(temp[z])) {
                        // console.log(curS.toLowerCase());
                        // console.log(pronounsArr[q]);
                        curS = "";
                    }
                }
            }

            //filter for randomFilterArray
            // for (var q = 0; q < randomFilterArray.length; q++) {
            //     if (curS.toLowerCase().includes(pronounsArr[q])) {
            //         curS = "";
            //     }
            // }

            curS = curS.trim();

            if (curS.length > 10) {
                var c = 0;
                var firstWord = curS.split(" ")[c];

                // var lastWord = curS.split(" ")[curS.split(" ").length - 1].toLowerCase();

                //why is the blank thing not working?
                curS += "."
                // var verb = await isVerb(firstWord);

                var verb = capitalized(firstWord);
                // console.log(verb);

                if (!verb) {

                    if (firstWord[firstWord.length - 1] == ":") {
                        //remove subheading that are characterized by a colon
                        curS = curS.slice(firstWord.length, curS.length);
                    }

                    // if (lastWord == "fig.") {
                    //     curS = curS.slice(0, curS.length - lastWord.length);
                    // }

                    console.log(curS + "\n");
                    sentenceArray.push(curS);
                }


            }


        }

        var result = "";
        for (var j = 0; j < sentenceArray.length; j++) {
            result += sentenceArray[j];
        }

        fs.writeFileSync("./texts/regexed.txt", result);

    }

}

test();

