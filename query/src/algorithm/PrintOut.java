package algorithm;

import java.io.FileNotFoundException;
import java.io.PrintStream;

public class PrintOut {
    public static void main(String arg[]) {
        PrintOut p = new PrintOut();// 构造对象
        System.out.print("Reallly?");
        System.out.println("Yes");
        System.out.println("So easy");



    }
    public PrintOut(){
        try {

            PrintStream print=new PrintStream(System.getProperty("user.dir") + "/printInfo/printInfo.txt");  //写好输出位置文件；
            System.setOut(print);
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }
    }
}

