

const express =  require("express")
const app = express();

app.use(express.json());

app.get("/hello" , (req , res) => {
    res.send("Hello from habit tracker!")
});

app.get("/api/status" , (req , res) =>
{
    res.json(
        name = "Habit tracker api",
        version = "1.0.0",
        status = "running"    )
})

app.listen(5000 , () => {
    console.log("Server is runnning")
})