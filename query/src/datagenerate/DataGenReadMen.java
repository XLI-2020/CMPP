package datagenerate;

import indoor_entitity.Door;
import indoor_entitity.Floor;
import indoor_entitity.IndoorSpace;
import indoor_entitity.Partition;
import utilities.RoomType;

import java.io.FileWriter;
import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Iterator;
import java.util.Scanner;

public class DataGenReadMen {

    public static String inputDoor = System.getProperty("user.dir") + "/inputfiles/Menzies/door.txt";
    public static String inputRoom = System.getProperty("user.dir") + "/inputfiles/Menzies/room.txt";

    public void dataGen() throws IOException {
        Path path1 = Paths.get(inputRoom);
        Scanner scanner1 = new Scanner(path1);

        while (scanner1.hasNextLine()) {
            String line = scanner1.nextLine();
            String [] tempArr = line.split(" ");
            int roomId = Integer.parseInt(tempArr[0]);
            int roomType;
            if (tempArr[1] == "ROOM") {
                roomType = RoomType.STORE;
            }
            else if (tempArr[1] == "STAIR") {
                roomType = RoomType.STAIRCASE;
            }
            else {
                roomType = RoomType.HALLWAY;
            }

            double x1 = Double.parseDouble(tempArr[2]);
            double y1 = Double.parseDouble(tempArr[3]);
            double x2 = Double.parseDouble(tempArr[4]);
            double y2 = Double.parseDouble(tempArr[5]);
            int floor = Integer.parseInt(tempArr[6]) + 1;
            String [] doorArr = tempArr[7].split(",");

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

            Partition par = new Partition(x1, x2, y1, y2, roomType);
            par.setmFloor(floor);

            for (int i = 0; i < doorArr.length; i++) {
                int doorId = Integer.parseInt(doorArr[i]);
                par.addDoor(doorId);
            }
            IndoorSpace.iPartitions.add(par);
        }

        Path path2 = Paths.get(inputDoor);
        Scanner scanner2 = new Scanner(path2);

        while (scanner2.hasNextLine()) {
            String line = scanner2.nextLine();
            String [] tempArr = line.split(" ");
            int doorId = Integer.parseInt(tempArr[0]);
            int room1  = Integer.parseInt(tempArr[1]);
            int room2 = Integer.parseInt(tempArr[2]);
            double x = Double.parseDouble(tempArr[3]);
            double y = Double.parseDouble(tempArr[4]);
            Door door = new Door(x, y);

            door.setmPartitions(new ArrayList<Integer>(Arrays.asList(room1, room2)));
            IndoorSpace.iDoors.add(door);

        }

        for (int i = 0; i < 17; i++) {
            Floor floor = new Floor(i);
            for (int j = 0; j < IndoorSpace.iPartitions.size(); j++) {
                Partition par = IndoorSpace.iPartitions.get(j);
                if (par.getmFloor() == i) {
                    floor.addPartition(par.getmID());
                }
            }
            IndoorSpace.iFloors.add(floor);
        }
    }
    private boolean saveDP() {
        try {
            FileWriter fwPar = new FileWriter(System.getProperty("user.dir") + "/inputfiles/Par_new.txt");
            Iterator<Partition> itrPar = IndoorSpace.iPartitions.iterator();
            while (itrPar.hasNext()) {
                fwPar.write(itrPar.next().toString() + "\n");
            }
            fwPar.flush();
            fwPar.close();
        } catch (IOException e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
            return false;
        }

        try {
            FileWriter fwDoor = new FileWriter(System.getProperty("user.dir") + "/inputfiles/Door_new.txt");
            Iterator<Door> itrDoor = IndoorSpace.iDoors.iterator();
            while (itrDoor.hasNext()) {
                fwDoor.write(itrDoor.next().toString() + "\n");
            }
            fwDoor.flush();
            fwDoor.close();
        } catch (IOException e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
            return false;
        }
        return true;
    }
}

