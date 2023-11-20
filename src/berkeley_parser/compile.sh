javac edu/berkeley/nlp/PCFGLA/BerkeleyParser.java -d compiled/
jar -cfm CustomBerkeley.jar META-INF/MANIFEST.MF -C compiled/ .
