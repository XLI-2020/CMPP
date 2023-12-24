package algorithm;
import com.google.gson.Gson;

import java.util.HashMap;


import com.google.gson.stream.JsonReader;

import java.io.FileReader;
import java.io.IOException;
import java.util.Set;


public class CalProbability {
    public static HashMap<String, com.google.gson.internal.LinkedTreeMap> jsonMap;
    public static HashMap<Double, HashMap<Double, Double>> normMap = new HashMap<>();

    public static void jsonConvert() throws IOException {
        JsonReader getLocalJsonFile = new JsonReader(new FileReader(System.getProperty("user.dir") + "/norm_distribution/norm_distribution.json"));

        jsonMap = new Gson().fromJson(getLocalJsonFile, HashMap.class);
        Set<String> key1s = jsonMap.keySet();

        for (String key1: key1s) {
            com.google.gson.internal.LinkedTreeMap  value1 = jsonMap.get(key1);
            HashMap<Double, Double> temp = new HashMap<>();
            Set<Object> key2s = value1.keySet();
            for (Object key2: key2s) {
                Object value2 = value1.get(key2);
                temp.put(Double.parseDouble((String)key2), (double)value2);
            }
            normMap.put(Double.parseDouble(key1), temp);


        }

    }

    public static double calProbability(int population, double a, double u) {
        double probability = 0;
        double norm = (population - a) / Math.sqrt(u);
//        System.out.println("norm: " + norm);
        if (norm < 0) {
            if (norm <= -4) {
                probability = 1;
                return probability;
            }
            else {
                double temp1 = (double) Math.round(Math.abs(norm) * 100) / 100;
//                System.out.println("temp1: " + temp1);
                double temp = temp1 * 10;
//                System.out.println("temp: " + temp);
                int i = (int)temp;
//                System.out.println("i: " + i);
                double d = Math.abs(temp - i);
//                System.out.println("d: " + d);
                double key1 = (double)i / 10;
                double key2 = (double) Math.round((d / 10) * 100) / 100;
//                System.out.println("key1: " + key1 + ", key2: " + key2);
                probability = normMap.get(key1).get(key2);
//                System.out.println("probability: " + probability);
                return probability;

            }

        }
        else {
            if (norm >= 4) {
                probability = 0;
                return probability;
            }
            else {
                double temp1 = (double) Math.round(Math.abs(norm) * 100) / 100;
//                System.out.println("temp1: " + temp1);
                double temp = temp1 * 10;
                int i = (int)temp;
//                System.out.println("i: " + i);
                double d = Math.abs(temp - i);
//                System.out.println("d: " + d);
                double key1 = (double)i / 10;
                double key2 = (double) Math.round((d / 10) * 100) / 100;
//                System.out.println("key1: " + key1 + ", key2: " + key2);
                probability = 1 - normMap.get(key1).get(key2);
                return probability;
            }
        }
//        return probability;
    }

    public static void main(String[] arg) throws IOException {
        jsonConvert();

        calProbability(5, 10, 5);
    }
}

