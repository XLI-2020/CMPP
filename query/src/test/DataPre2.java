package test;

import indoor_entitity.Door;
import indoor_entitity.IndoorSpace;
import indoor_entitity.Partition;

import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Scanner;

public class DataPre2 {
    public static String inputDoor_new = System.getProperty("user.dir") + "/inputfiles/map_to_txt_data/door.txt";
    public static String inputRoom_new = System.getProperty("user.dir") + "/inputfiles/map_to_txt_data/par.txt";

    public static String outputDoor = System.getProperty("user.dir") + "/inputfiles/map_to_txt_data/door_final.txt";
    public static String outputPar = System.getProperty("user.dir") + "/inputfiles/map_to_txt_data/par_final.txt";

    public static String inputDoorIdInfo = System.getProperty("user.dir") + "/inputfiles/map_to_txt_data/doorIDInfo.txt";
    public static String inputParIdInfo = System.getProperty("user.dir") + "/inputfiles/map_to_txt_data/parIDInfo.txt";

    public static HashMap<Integer, Integer> parIds = new HashMap<>(); // key: original Id; value: new Id;

    public static void main(String[] args) throws IOException {
        findParId();
        readDoor();
        readPar();
        save();

    }

    public static void findParId() throws IOException {
        Path path = Paths.get(inputParIdInfo);
        Scanner scanner = new Scanner(path);

        while (scanner.hasNextLine()) {
            String line = scanner.nextLine();
            String [] tempArr = line.split("\t");
            int originalId = Integer.parseInt(tempArr[1]);
            int newId = Integer.parseInt((tempArr[0]));
            parIds.put(originalId, newId);
        }
    }

    public static void readDoor() throws IOException {
        Path path2 = Paths.get(inputDoor_new);
        Scanner scanner2 = new Scanner(path2);
        while (scanner2.hasNextLine()) {
            String line = scanner2.nextLine();
            String[] tempArr = line.split("\t");
            int doorId = Integer.parseInt(tempArr[0]);
//            System.out.println("doorId: " + doorId);
            double x = Double.parseDouble(tempArr[1]);
            double y = Double.parseDouble(tempArr[2]);
            int floor = Integer.parseInt(tempArr[3]);

            Door door = new Door(x, y);

            String[] parArr = tempArr[4].split(",");
            ArrayList<Integer> parList = new ArrayList<>();
            for (int i = 0; i < parArr.length; i++) {
                int parId_ori = Integer.parseInt(parArr[i]);
                int parId = parIds.get(parId_ori);
                parList.add(parId);
            }

            door.setmFloor(floor);
            door.setmPartitions(parList);

            IndoorSpace.iDoors.add(door);

        }


    }

    public static void readPar() throws IOException {
        Path path = Paths.get(inputRoom_new);
        Scanner scanner = new Scanner(path);

        while (scanner.hasNextLine()) {
            String line = scanner.nextLine();
            String[] tempArr = line.split("\t");
            int parId_ori = Integer.parseInt(tempArr[0]);
//            System.out.println("doorId: " + doorId);
            double x1 = Double.parseDouble(tempArr[1]);
            double y1 = Double.parseDouble(tempArr[2]);
            double x2 = Double.parseDouble(tempArr[3]);
            double y2 = Double.parseDouble(tempArr[4]);
            int floor = Integer.parseInt(tempArr[5]);
            int type = Integer.parseInt(tempArr[6]);

            if (x1 > x2) {
                double temp = x2;
                x2 = x1;
                x1 = temp;
            }

            if (y1 > y2) {
                double temp = y2;
                y2 = y1;
                y1 = temp;
            }

            Partition par = new Partition(x1, x2, y1, y2, type);
            par.setmFloor(floor);
            IndoorSpace.iPartitions.add(par);

        }

        for (int i = 0; i < IndoorSpace.iDoors.size(); i++) {
//            System.out.println("doorId " + i);
            Door door = IndoorSpace.iDoors.get(i);
            ArrayList<Integer> parIds = door.getmPartitions();
            for (int j = 0; j < parIds.size(); j++) {
                Partition par = IndoorSpace.iPartitions.get(parIds.get(j));
                if (!par.getmDoors().contains(door.getmID())) {
                    par.addDoor(door.getmID());
                }
            }
        }


    }

    public static void save() throws IOException {
        String resultDoor = "";
        String resultPar = "";

        for (int i = 0; i < IndoorSpace.iDoors.size(); i++) {
            Door door = IndoorSpace.iDoors.get(i);
            resultDoor += door.toString() + "\n";
        }

        for (int i = 0; i < IndoorSpace.iPartitions.size(); i++) {
            Partition par = IndoorSpace.iPartitions.get(i);
            resultPar += par.toString() + "\n";
        }

        FileOutputStream output1 = new FileOutputStream(outputDoor);
        output1.write(resultDoor.getBytes());
        output1.flush();
        output1.close();

        FileOutputStream output2 = new FileOutputStream(outputPar);
        output2.write(resultPar.getBytes());
        output2.flush();
        output2.close();
    }

}
