package com.fyp.PDFBox;

import org.apache.pdfbox.contentstream.operator.Operator;
import org.apache.pdfbox.cos.COSName;
import org.apache.pdfbox.pdfparser.PDFStreamParser;
import org.apache.pdfbox.pdfwriter.ContentStreamWriter;
/**
 * Hello world!
 *
 */
import org.apache.pdfbox.pdmodel.PDDocument;
import org.apache.pdfbox.text.PDFTextStripper;
import org.apache.pdfbox.pdmodel.PDPage;
import org.apache.pdfbox.pdmodel.PDResources;
import org.apache.pdfbox.pdmodel.common.PDStream;
import org.apache.pdfbox.pdmodel.graphics.PDXObject;
import org.apache.pdfbox.pdmodel.graphics.image.PDImageXObject;
import org.apache.pdfbox.pdmodel.graphics.pattern.PDAbstractPattern;

import java.io.File;
import java.io.IOException;
import java.io.OutputStream;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class App 
{
	public static void removeImages(String pdfFile) throws Exception {
        PDDocument document = PDDocument.load(new File(pdfFile));

        for (PDPage page : document.getPages()) {
            PDResources pdResources = page.getResources();
            pdResources.getXObjectNames().forEach(propertyName -> {
                if(!pdResources.isImageXObject(propertyName)) {
                    return;
                }
                PDXObject o;
                try {
                    o = pdResources.getXObject(propertyName);
                    if (o instanceof PDImageXObject) {
                        System.out.println("propertyName" + propertyName);
                        page.getCOSObject().removeItem(propertyName);
                    }
                } catch (IOException e) {
                    e.printStackTrace();
                }
            });

            for (COSName name :  page.getResources().getPatternNames()) {
                PDAbstractPattern pattern = page.getResources().getPattern(name);
                System.out.println("have pattern");
            }
              

            PDDocument newDoc = new PDDocument();
            PDPage newPage = newDoc.importPage(page);
            newPage.setResources(page.getResources());

            PDStream newContents = new PDStream(newDoc);
            newPage.setContents( newContents );
        }

//        document.save("RemoveImage.pdf");
//        document.close();
        
        PDFTextStripper pdfStripper = new PDFTextStripper();
        String text = pdfStripper.getText(document);
        System.out.println(text);
        
        document.close();
    }
	
    public static void main( String[] args ) throws Exception
    {
        File pdfFile = new File(args[0]);
        PDDocument document = PDDocument.load(pdfFile);
        
//        document.save("C:\\PDF\\myPDF.pdf");
//        System.out.println("PDF Created");
        
        PDFTextStripper pdfStripper = new PDFTextStripper();
        String text = pdfStripper.getText(document);
//        PDResources resources = null;
//        for (PDPage page : document.getPages()) {
//            resources = page.getResources();
//
//            for (COSName name : resources.getXObjectNames()) {
//                PDXObject xobject = resources.getXObject(name);
//                
//                if (xobject instanceof PDImageXObject) {
//                    System.out.println("have image");
//                    removeImages("C:\\PDF\\Chapter2.pdf");
//                }
//            }
//        }
        
//        Map<Character, Character> replacements = new HashMap<>();
//        replacements.put('\n', ' ');
//        
//        StringBuilder output = new StringBuilder();
//        for (Character c : text.toCharArray()) {
////            output.append(replacements.getOrDefault(c,c));
//        	if(c != '\n') {
//        		output.append(c);
//        	}
////        	else {
////        		output.append("\\");
////        	}
//        }
        
//        System.out.print(output);
        
        System.out.println(text);
        
        
        document.close();
        
    }
}
