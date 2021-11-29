package test;

import utilities.RoomType;

import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Scanner;

public class DataPre {
    public static String inputDoor_new = System.getProperty("user.dir") + "/inputfiles/map_to_txt_data/doors_info_floor_7.txt";
    public static String inputRoom_new = System.getProperty("user.dir") + "/inputfiles/map_to_txt_data/partitions_info_floor_7.txt";

    public static String outputDoor = System.getProperty("user.dir") + "/inputfiles/map_to_txt_data/door.txt";
    public static String outputPar = System.getProperty("user.dir") + "/inputfiles/map_to_txt_data/par.txt";

    public static String outputDoorIdInfo = System.getProperty("user.dir") + "/inputfiles/map_to_txt_data/doorIDInfo.txt";
    public static String outputParIdInfo = System.getProperty("user.dir") + "/inputfiles/map_to_txt_data/parIDInfo.txt";

    public static void main(String[] args) throws IOException {
//        readDoor();
        readPar();

    }

    public static void readDoor() throws IOException {
        String doorInfo = "";
        String doorIdInfo = "";
        Path path2 = Paths.get(inputDoor_new);
        Scanner scanner2 = new Scanner(path2);

        int doorNum = 1313;
        int floorNum = 6;
        while (scanner2.hasNextLine()) {
            String line = scanner2.nextLine();
            String[] tempArr = line.split("\t");
            int doorId_ori = Integer.parseInt(tempArr[1]);
//            System.out.println("doorId: " + doorId);
            double x = Double.parseDouble(tempArr[2]);
            double y = Double.parseDouble(tempArr[3]);

            String[] parArr = tempArr[4].split(", ");
            String parList = "";
            if (parArr.length != 2) {
                if (parArr.length < 2) {
                    int parId1 = Integer.parseInt(parArr[0]) + floorNum * 1000;
                    parList = parId1 + "";
                    System.out.println("partition list less than 2 " + doorNum);
//                    continue;
                }
                if (parArr.length > 2) {
                    int parId1 = Integer.parseInt(parArr[0]) + floorNum * 1000;
                    int parId2 = Integer.parseInt(parArr[1]) + floorNum * 1000;
                    int parId3 = Integer.parseInt(parArr[2]) + floorNum * 1000;
                    parList = parId1 + "," + parId2 + "," + parId3;
                    System.out.println("partition list greater than 2 " + doorNum);
//                    continue;
                }
            }
            else {

                int parId1 = (Integer.parseInt(parArr[0])) + floorNum * 1000;
                int parId2 = (Integer.parseInt(parArr[1])) + floorNum * 1000;
                parList = parId1 + "," + parId2;
            }

            doorIdInfo += doorNum + "\t" + (doorId_ori + floorNum * 1000) + "\n";
            doorInfo += (doorNum++) + "\t" + x + "\t" + y + "\t" + floorNum + "\t" + parList + "\n";


        }

        FileOutputStream output1 = new FileOutputStream(outputDoor, true);
        output1.write(doorInfo.getBytes());
        output1.flush();
        output1.close();

        FileOutputStream output2 = new FileOutputStream(outputDoorIdInfo, true);
        output2.write(doorIdInfo.getBytes());
        output2.flush();
        output2.close();


    }

    public static void readPar() throws IOException {
        String parInfo = "";
        String parIdInfo = "";
        Path path = Paths.get(inputRoom_new);
        Scanner scanner = new Scanner(path);

        int parNum = 824;
        int floorNum = 6;
        while (scanner.hasNextLine()) {
            String line = scanner.nextLine();
            String[] tempArr = line.split("\t");
            int parId_ori = Integer.parseInt(tempArr[1]);
//            System.out.println("doorId: " + doorId);
            double x1 = Double.parseDouble(tempArr[2]);
            double y1 = Double.parseDouble(tempArr[3]);
            double x2 = Double.parseDouble(tempArr[4]);
            double y2 = Double.parseDouble(tempArr[5]);
            String type = tempArr[6];
            int parType = -1;
            if (type.equals("hallway")) {
                parType = RoomType.HALLWAY;
            }
            else if (type.equals("escalator") || type.equals("staircase") || type.equals("lift")) {
                parType = RoomType.STAIRCASE;
            }
            else if (type.equals("shop") || type.equals("facility_room")) {
                parType = RoomType.STORE;
            }
            else {
                System.out.println("something wrong with the partition type " + parNum);
            }


//            String[] doorArr = tempArr[7].split(",");
//
//            int parId1 = Integer.parseInt(parArr[0]);
//            int parId2 = Integer.parseInt(parArr[1]);

            parIdInfo += parNum + "\t" + (parId_ori + floorNum * 1000) + "\n";
            parInfo += (parNum++) + "\t" + x1 + "\t" + y1 + "\t" + x2 + "\t" + y2 + "\t" + floorNum + "\t" + parType + "\n";


        }

        FileOutputStream output1 = new FileOutputStream(outputPar, true);
        output1.write(parInfo.getBytes());
        output1.flush();
        output1.close();

        FileOutputStream output2 = new FileOutputStream(outputParIdInfo, true);
        output2.write(parIdInfo.getBytes());
        output2.flush();
        output2.close();


    }

}
