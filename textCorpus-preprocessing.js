import fs from "fs";
// var toJSON = require('plain-text-data-to-json')
import wd from "word-definition";
import { capitalized } from "is-verb";
import pos from "pos";

import wordToNum from "word-to-numbers";

function isVerb(firstWord) {
    //returns 1 if verb else 0 if not
    var ans = 0;
    const x = wd.getDef(firstWord, "en", null, function (definition) {
        // console.log(definition.category);
        if (definition.category == "verb") {
            console.log("hi");
            ans = 1;
        }
    });

    return ans;
}

function containsProperNoun(sentence) {
  var words = new pos.Lexer().lex(sentence);
  var tagger = new pos.Tagger();
  var taggedWords = tagger.tag(words);
  // console.log(taggedWords);
  var i = 0;
  for (i in taggedWords) {
    var taggedWord = taggedWords[i];
    var word = taggedWord[0];
    var tag = taggedWord[1];
    // if(tag==="NNP" || tag ==="NNPS"){
    //     return true;
    // }
  }

  return false;
}

async function textcorpusPreprocessing() {

  var title = "directFromPDFWithoutNewline";
  var doc = fs.readFileSync("./texts/" + title + ".txt", 'utf8')

  var sentenceArray = [];

  var doc1 = ".";

  doc1 += doc;

  doc1 = doc1.replace(/ [1-9]+[.][1-9]+ /g, decimalReplacer); // deal with decimals prior

  // doc1 = doc1.replace(/ [.][ ]([1-9]+[.][1-9]+[ ]) /g, replacer); // deal with decimals prior

  function decimalReplacer(match, p1, p2, p3, offset, string) {
    // p1 is non-digits, p2 digits, and p3 non-alphanumerics
    let x = match;
    let y = Math.trunc(Number(x));
    // console.log("match:" + x);
    return " around " + y + " ";
    // return " " + y + " ";
  }

  //loop through the whole doc and convert words into numbers
  var wordsInText = doc1.split(" ");
  // console.log(wordsInText);

  var i = 0;
  for (i in wordsInText) {
    let x = wordsInText[i].trim();
    let t = wordToNum(x);
    //console.log(t + "\n");
    if (Number.isInteger(t)) {
      doc1 = doc1.replace(x, t);
    }
  }

  doc1 = doc1.replace(/ [1-9]+ lakhs/g, wordToNumberReplacer);
  doc1 = doc1.replace(/ [1-9]+ lakh/g, wordToNumberReplacer);
  doc1 = doc1.replace(/ [1-9]+ crores/g, wordToNumberReplacer);
  doc1 = doc1.replace(/ [1-9]+ crore/g, wordToNumberReplacer);
  doc1 = doc1.replace(/ [1-9]+ thousands/g, wordToNumberReplacer);
  doc1 = doc1.replace(/ [1-9]+ thousand/g, wordToNumberReplacer);

  function wordToNumberReplacer(match, p1, p2, p3, offset, string) {
    // p1 is non-digits, p2 digits, and p3 non-alphanumerics
    let wholeString = match;

    let noPart = "";
    wholeString = wholeString.trim();

    for (var i = 0; i < wholeString.length; i++) {
      if (Number(wholeString[i]) > 1 && Number(wholeString[i]) <= 9) {
        noPart += wholeString[i];
      } else {
        break;
      }
    }

    let multiplier = wholeString
      .slice(noPart.length + 1, wholeString.length)
      .trim();

    noPart = Number(noPart);

    if (multiplier === "crore" || multiplier === "crores") {
      noPart *= 10000000;
    } else if (multiplier === "lakh" || multiplier === "lakhs") {
      noPart *= 100000;
    } else if (multiplier === "thousand" || multiplier === "thousands") {
      noPart *= 1000;
    }

    // console.log(noPart);
    return " " + noPart + " ";
  }

  doc1 = doc1.replace(/ *\([^)]*\) */g, ""); //remove brackets
  doc1 = doc1.replace(/Q 2022-23/g, "");
  doc1 = doc1.replace(/2022-23/g, "");
  doc1 = doc1.replace(/Fig/g, "");
  doc1 = doc1.replace(/[A-Z]+[-]*[A-Z]+[ ]*[1-9]*[:]*/g, "");
  doc1 = doc1.replace(/[�]*/g, "");

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
        if (sentenceLetterCount < 4 && Number(doc1[i])) {
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

      var pronounsArr = ["he", "she", "his", "they", "us", "you"];

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

      // if(!containsProperNoun(curS)){
      //     curS = "";
      // }

      var firstWord = curS.split(" ")[c];

      if (firstWord === "around") {
        curS = curS.slice(firstWord.length + 3, curS.length);
      }

      if (curS.length > 20) {
        var c = 0;
        var firstWord = curS.split(" ")[c];

        // var lastWord = curS.split(" ")[curS.split(" ").length - 1].toLowerCase();

        //why is the blank thing not working?
        curS += ".";
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
          if (curS[0] === ",") {
            curS = curS.slice(1, curS.length);
          }
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

textcorpusPreprocessing();
